#!/usr/bin/python3
from __future__ import print_function
import json
import struct
import bz2
import enum
from ctypes import *
import io
import sys
import os
from operator import attrgetter
import functools
from collections import defaultdict, namedtuple
from typing import List,Union

import hashlib
import re

GSF_PROF_RAW = 'gsf-profile.raw'

ACC_FAMILY = {0: 'VMP', 1: 'MPC', 2: 'PMA', 3: 'PMAC'} # on eq6 3: 'XNN'}
GSF_SEP_GOAL, GSF_START_GOAL, GSF_SYNC_GOAL, GSF_MSG_GOAL, GSF_VMP_GOAL = range(1, 6)
GSF_FOR_TASK, GSF_VMP_PROC_TASK, GSF_WAIT_TASK, GSF_SYNC_TASK, GSF_START_TASK, GSF_OCL_KERNEL_TASK = range(1, 7) # must keep GSF_OCL_KERNEL_TASK for backward compatibility
class PLL(enum.Enum):
  CPU, DDR0, DDR1, MPC, PCI, PER, PMA, PMAC, VDI, VMP, COUNT = range(0, 11)

goal_type_names = {
  GSF_SEP_GOAL:'GSF_SEP_GOAL',
  GSF_START_GOAL:'GSF_START_GOAL',
  GSF_SYNC_GOAL:'GSF_SYNC_GOAL',
  GSF_MSG_GOAL:'GSF_MSG_GOAL',
  GSF_VMP_GOAL:'GSF_VMP_GOAL'
}
task_type_names = {
  GSF_FOR_TASK:'GSF_FOR_TASK',
  GSF_VMP_PROC_TASK:'GSF_VMP_PROC_TASK',
  GSF_WAIT_TASK:'GSF_WAIT_TASK',
  GSF_SYNC_TASK:'GSF_SYNC_TASK',
  GSF_START_TASK:'GSF_START_TASK',
  GSF_OCL_KERNEL_TASK:'GSF_OCL_KERNEL_TASK'
}

class GoalType(enum.IntEnum):
  GSF_SEP_GOAL = GSF_SEP_GOAL
  GSF_START_GOAL = GSF_START_GOAL
  GSF_SYNC_GOAL = GSF_SYNC_GOAL
  GSF_MSG_GOAL = GSF_MSG_GOAL
  GSF_VMP_GOAL = GSF_VMP_GOAL

class TaskType(enum.IntEnum):
  GSF_FOR_TASK = GSF_FOR_TASK
  GSF_VMP_PROC_TASK = GSF_VMP_PROC_TASK
  GSF_WAIT_TASK = GSF_WAIT_TASK
  GSF_SYNC_TASK = GSF_SYNC_TASK
  GSF_START_TASK = GSF_START_TASK
  GSF_OCL_KERNEL_TASK = GSF_OCL_KERNEL_TASK

def _magic_num(s):
  return struct.unpack('I',s)[0]

def print_struct(s):
  print('%s(%s)' % (s.__class__.__name__, ', '.join('{}={}'.format(name, getattr(s, name)) for name, type in s._fields_)))

# should match the structures defined in gsf_prof.cpp
class prof_header(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('version',c_int)]
  MAGIC = _magic_num(b'GSFp')

class prof_app_section(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('size',c_int), ('app_id',c_uint)]
  COMPRESSED_MAGIC = _magic_num(b'APPp')
  UNCOMPRESSED_MAGIC = _magic_num(b'AJSp')
  MAGIC = [COMPRESSED_MAGIC, UNCOMPRESSED_MAGIC]

class eyeq_cfg_section_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('pll',c_uint64 * PLL.COUNT.value)]
  MAGIC = _magic_num(b'CFGp')

  def __getattr__(self, name): # sort helpers
  # temp hack until gsf dumps rev
  #       0   1    2    3   4   5   6   7    8   9
  # 5 = { cpu,ddr0,ddr1,mpc,pci,per,pma,pmac,vdi,vmp" };
  # 6 = { acc,ddr, ddr, acc,per,per,acc,acc, vdi,acc }
    if name  == 'eyeq_rev':
      if  all(map(lambda x:self.pll[x] == self.pll[0], [0,3,6,7,9])):
        return 6
      else:
        return 5

class eyeq_cfg_section_v9(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('rev',c_int8), ('rev_str_arr',c_char*11), ('pll',c_uint64 * PLL.COUNT.value)]
  MAGIC = _magic_num(b'CFGp')

  def __getattr__(self, name): # sort helpers
    if name  == 'eyeq_rev':
      return self.rev
    elif name == 'rev_str':
      return self.rev_str_arr.decode()

class prof_run_section_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('size',c_int), ('frame',c_int), ('app_id',c_uint)]
  MAGIC = _magic_num(b'RUNp')
  start = None
  finish = None

class prof_run_section_v8(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('size',c_int), ('frame',c_int), ('app_id',c_uint), ('start', c_ulonglong), ('finish', c_ulonglong)]
  MAGIC = _magic_num(b'RUNp')

class prof_dt_section(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('size',c_int)]
  MAGIC = _magic_num(b'DYNp')

class prof_dep_section(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_int), ('size',c_int), ('index',c_int)]
  MAGIC = _magic_num(b'DEPp')

class dynamic_task_info(object):
  def __init__(self, name, type):
    self.name = name
    self.type = type
  MAGIC = ord('D')
  t2s = {GSF_FOR_TASK:'GSF_FOR_TASK', GSF_VMP_PROC_TASK:'GSF_VMP_PROC_TASK', GSF_WAIT_TASK:'GSF_WAIT_TASK'}

  def __str__(self):
    return self.name + ' ' + self.t2s[self.type]

def check_magic(header, struct_type):
  try:
    return header.magic in struct_type.MAGIC
  except TypeError:
    return header.magic == struct_type.MAGIC

def magic_name(struct_type):
  try:
    return u','.join(struct.pack('I', magic).decode('utf-8') for magic in struct_type.MAGIC)
  except TypeError:
    return struct.pack('I', struct_type.MAGIC).decode('utf-8')

# profiling records:
magic2level = {'G': 0, 'T': 1, 'A': 2, 'F': 2, 'D': 2}
def run_record(magic):
  def go(cls):
    if not hasattr(cls, 'type'):
      cls.type = None
    cls.nesting_level = magic2level[magic]
    fields = {name for name, type in cls._fields_}
    assert cls._fields_[0] == ('magic', c_uint8)
    if cls.nesting_level == 0:
      cls.gindex = property(lambda self: self.index)
      cls.full_index = property(lambda self: self.index)
      if '_type' not in fields: cls._type = None
      cls.type = property(lambda self: GoalType(self._type) if self._type is not None else None,
                          (lambda self, value: setattr(self, '_type', value)))
    elif cls.nesting_level == 1:
      cls.tindex = property(lambda self: self.index)
      cls.full_index = property(lambda self: (self.gindex, self.index))
      if 'gindex' not in fields: cls.gindex = None
      if '_type' not in fields: cls._type = None
      cls.type = property(lambda self: TaskType(self._type) if self._type is not None else None,
                          (lambda self, value: setattr(self, '_type', value)))
      cls.dep = None
    else:
      cls.full_index = property(lambda self: (self.gindex, self.tindex, self.index))
      if 'gindex' not in fields: cls.gindex = None
      if 'tindex' not in fields: cls.tindex = None
    mint = ord(magic)
    oldinit = cls.__init__
    def newinit(self, *args, **kwargs):
      if len(args) > 0:
        assert args[0] == mint
      elif 'magic' in kwargs:
        assert kwargs['magic'] == mint
      else:
        kwargs['magic'] = mint
      return oldinit(self, *args, **kwargs)
    return cls
  return go

@run_record('G')
class goal_prof_record_v3(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('index',c_uint16), ('_type',c_uint8), ('tid',c_int), ('cid',c_uint8), ('start',c_uint64), ('finish',c_uint64)]
  def __str__(self):
    return "G %d %d %u %d %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.finish)

@run_record('T')
class task_prof_record_v3(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('index',c_uint16), ('_type',c_uint8), ('tid',c_int), ('cid',c_uint8), ('start',c_uint64), ('suspend',c_uint64), ('resume',c_uint64), ('finish',c_uint64)]
  def __str__(self):
    return "T %d %d %u %d %u %u %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.suspend, self.resume, self.finish)

@run_record('A')
class accl_prof_record_v3(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('index',c_uint16), ('family',c_uint8), ('iid',c_uint8), ('start',c_uint64), ('finish',c_uint64)]
  def __str__(self):
    return "A %d %d %d %u %u" % (self.index, self.family, self.iid, self.start, self.finish)

@run_record('G')
class goal_prof_record_v4(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('_type',c_uint8), ('index',c_uint16), ('tid',c_int), ('start',c_uint64), ('finish',c_uint64), ('cid',c_uint8)] # TODO, ('pad',c_uint8*7)]
  nesting_level = 0 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "G %d %d %u %d %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.finish)

@run_record('T')
class task_prof_record_v4(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('_type',c_uint8), ('index',c_uint16), ('tid',c_int), ('start',c_uint64), ('suspend',c_uint64), ('resume',c_uint64), ('finish',c_uint64), ('proc_task',c_uint16), ('cid',c_uint8)]# TODO, ('pad',c_uint8*5)]
  nesting_level = 1 # 0 for goal, 1 for task, 2 for job
  gindex = None
  def __str__(self):
    return "T %d %d %u %d %u %u %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.suspend, self.resume, self.finish)

@run_record('A')
class accl_prof_record_v4(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('family', c_uint8), ('index', c_uint16), ('iid', c_uint8), ('start', c_uint64), ('finish', c_uint64)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  gindex = None
  tindex = None
  def __str__(self):
    return "A %d %d %d %u %u" % (self.index, self.family, self.iid, self.start, self.finish)

@run_record('G')
class goal_prof_record_v5(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('cid',c_uint8), ('index',c_uint16), ('tid',c_int), ('start',c_uint64), ('finish',c_uint64)]
  nesting_level = 0 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "G %d %s %u %d %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.finish)

@run_record('G')
class goal_prof_record_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('cid',c_uint8), ('index',c_uint16), ('tid',c_uint16), ('_pad',c_uint8*2), ('start',c_uint64), ('finish',c_uint64)]
  nesting_level = 0 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "G %u %s %u %d %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.finish)

@run_record('T')
class task_prof_record_v5(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('cid',c_uint8), ('index',c_uint16), ('tid',c_int), ('start',c_uint64), ('suspend',c_uint64), ('resume',c_uint64), ('finish',c_uint64)]
  nesting_level = 1 # 0 for goal, 1 for task, 2 for job
  gindex = None
  def __str__(self):
    return "T %d %s %u %d %u %u %u %u" % (self.index, self.type, self.tid, self.cid, self.start, self.suspend, self.resume, self.finish)

@run_record('T')
class task_prof_record_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic',c_uint8), ('cid',c_uint8), ('gindex',c_uint16), ('index',c_int16), ('tid',c_uint16), ('start',c_uint64), ('suspend',c_uint64), ('resume',c_uint64), ('finish',c_uint64)]
  nesting_level = 1 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "T %d %d %s %u %d %u %u %u %u" % (self.gindex, self.index, self.type, self.tid, self.cid, self.start, self.suspend, self.resume, self.finish)

@run_record('A')
class accl_prof_record_v5(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('family', c_uint8), ('index', c_uint16), ('iid', c_uint8), ('_pad',c_uint8*3), ('start', c_uint64), ('finish', c_uint64)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  gindex = None
  tindex = None
  def __str__(self):
    return "A %d %d %d %u %u" % (self.index, self.family, self.iid, self.start, self.finish)

@run_record('A')
class accl_prof_record_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('family', c_uint8), ('gindex', c_uint16), ('tindex', c_int16), ('index', c_uint8), ('iid', c_uint8), ('start', c_uint64), ('finish', c_uint64)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "A %d %d %d %d %d %u %u" % (self.gindex, self.tindex, self.index, self.family, self.iid, self.start, self.finish)

@run_record('F')
class pfor_prof_record_v6(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('cid', c_uint8), ('iters', c_uint16), ('_pad',c_uint8*4), ('start', c_uint64), ('finish', c_uint64)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  gindex = None
  tindex = None
  def __str__(self):
    return "F %d %u %u %u" % (self.cid, self.iters, self.start, self.finish)

@run_record('F')
class pfor_prof_record_v7(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('cid', c_uint8), ('iters', c_int16), ('gindex', c_uint16), ('tindex', c_int16), ('start', c_uint64), ('finish', c_uint64)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "F %d %d %d %u %u %u" % (self.gindex, self.tindex, self.cid, self.iters, self.start, self.finish)

@run_record('D')
class dep_prof_record(Structure):
  _pack_ = 1
  _fields_ = [('magic', c_uint8), ('pad', c_uint8), ('gindex', c_uint16), ('tindex', c_int16), ('dep_index', c_uint16)]
  nesting_level = 2 # 0 for goal, 1 for task, 2 for job
  def __str__(self):
    return "D {} {} {}".format(self.gindex, self.tindex, self.dep_index)

goal_prof_record = None
task_prof_record = None
accl_prof_record = None
pfor_prof_record = None
prof_run_section = prof_run_section_v7
eyeq_cfg_section = eyeq_cfg_section_v7

def init_record_versions(version):
  global goal_prof_record, task_prof_record, accl_prof_record, pfor_prof_record, prof_run_section, eyeq_cfg_section
  if version in (0x00002000, 0x00003000):
    goal_prof_record = goal_prof_record_v3
    task_prof_record = task_prof_record_v3
    accl_prof_record = accl_prof_record_v3
    if version == 0x00002000:
      return (1.5e9, 1.0)
    else:
      return (1.5e9, 1.5)
  elif version == 0x00004000:
    goal_prof_record = goal_prof_record_v4
    task_prof_record = task_prof_record_v4
    accl_prof_record = accl_prof_record_v4
    return (1.5e9, 1.5)
  elif version == 0x00005000:
    goal_prof_record = goal_prof_record_v5
    task_prof_record = task_prof_record_v5
    accl_prof_record = accl_prof_record_v5
    return (None, None)
  elif version == 0x00006000:
    goal_prof_record = goal_prof_record_v5
    task_prof_record = task_prof_record_v5
    accl_prof_record = accl_prof_record_v5
    pfor_prof_record = pfor_prof_record_v6
    return (None, None)
  elif version == 0x00007000:
    goal_prof_record = goal_prof_record_v7
    task_prof_record = task_prof_record_v7
    accl_prof_record = accl_prof_record_v7
    pfor_prof_record = pfor_prof_record_v7
    return (None, None)
  elif version == 0x00008000:
    goal_prof_record = goal_prof_record_v7
    task_prof_record = task_prof_record_v7
    accl_prof_record = accl_prof_record_v7
    pfor_prof_record = pfor_prof_record_v7
    prof_run_section = prof_run_section_v8
    return (None, None)
  elif version == 0x00009000:
    goal_prof_record = goal_prof_record_v7
    task_prof_record = task_prof_record_v7
    accl_prof_record = accl_prof_record_v7
    pfor_prof_record = pfor_prof_record_v7
    prof_run_section = prof_run_section_v8
    eyeq_cfg_section = eyeq_cfg_section_v9
    return (None, None)
  else:
    raise Exception('invalid file format, unsupported version:"%x"' % (version))

def sizeofany(t):
  return sizeof(t) if t else 0

def make_magic2record():
  return {ord('G'): (goal_prof_record, sizeofany(goal_prof_record)),
          ord('T'): (task_prof_record, sizeofany(task_prof_record)),
          ord('A'): (accl_prof_record, sizeofany(accl_prof_record)),
          ord('F'): (pfor_prof_record, sizeofany(pfor_prof_record)),
          ord('D'): (dep_prof_record,  sizeofany(dep_prof_record))}

TimeRange = namedtuple("TimeRange", [
  'start', 'end', 'name', 'line', 'module', 'pre', 'dep', 'type', 'record', 'priority'
])

Node = namedtuple("Node", ['name', 'ranges', 'children', 'record', 'start', 'end'])


def replaceInList(where, what, to):
  replaced = False
  for elem in what:
    try:
      where.remove(elem)
      replaced = True
    except ValueError:
      pass
  assert replaced
  where.append(to)

def mergeTR(a: TimeRange, b: TimeRange):
  assert min(a.end,b.end) == max(a.start,b.start)
  assert a in b.pre or a in b.dep
  assert b in a.pre or b in a.dep
  res = a._replace(start= min(a.start, b.start),
                   end=max(a.end, b.end),
                   dep=a.dep + b.dep,
                   pre=a.pre + b.pre)
  res.pre.remove(a if a in b.pre else b)
  res.dep.remove(a if a in b.dep else b)
  for p in res.pre:
    replaceInList(p.dep, (a,b), res)
  for d in res.dep:
    replaceInList(d.pre, (a,b), res)
  return res

def editTR(tr: TimeRange, **kwargs):
  r = tr._replace(**kwargs)
  for p in r.pre:
    p.dep[p.dep.index(tr)] = r
  for d in r.dep:
    d.pre[d.pre.index(tr)] = r
  return r

class DepKind(enum.Enum):
  ON_START, ON_END = range(2)
Dependency = namedtuple("Dependency", ['pre', 'dep', 'kind'])

def depend(pre, dep, kind: DepKind):
  if type(pre) is TimeRange and type(dep) is TimeRange:
    d = Dependency(pre, dep, kind)
    pre.dep.append(dep)
    dep.pre.append(pre)
    return d
  elif type(pre) is TimeRange:
    return [depend(pre, d, kind) for d in dep]
  elif type(dep) is TimeRange:
    return [depend(p, dep, kind) for p in pre]
  else:
    return [depend(p, d, kind) for p in pre for d in dep]


def extract_task_subname(goal_name: str, task_name: str, wait):
    prefix = goal_name + "_task"
    assert task_name.startswith(prefix)
    name = task_name[len(prefix):].lstrip("1234567890")
    if " [" in name:
        name = name[:name.index(" [")]
    return name

class GSFProfSource:
  def __init__(self, path):
    self.is_clip = self.file_or_clip(path)
    if self.is_clip:
      self.init_clip(path)
    else:
      self.file = open(path, 'rb')
      assert self.file, 'error opening :"%s"' % path
      self.file.seek(0, os.SEEK_END)
      self.fsize = self.file.tell()
      self.file.seek(0, os.SEEK_SET)

  def file_or_clip(self, path):
    if os.path.isdir(path) or path.endswith('.emp4'): # TODO: more possible exts?
      return True
    if os.path.basename(path) == GSF_PROF_RAW or path.endswith('.raw'):
      return False
    # TODO: refine conditions
    return False

  def init_clip(self, path):
    sys.path.append('/mobileye/EPG_stack/epmfarm/apps/')
    from clipReader_wrapper import CrInterface
    DEFAULT_CLIP_READER_LIB = '/homes/swlab/deliver/sdk/curr/bin/lin64/libclipread.so'

    try:
      self.cr = CrInterface(path, DEFAULT_CLIP_READER_LIB, "[Clip]\nopenMode=reader\n")
    except:
      raise Exception("error initializing clipreader:'%s'" % path)

    assert 0 == self.cr.open_clip(), "error opening clip:'%s'" % path
    assert 0 == self.cr.clip_info(), "error getting clip info:'%s'" % path
    self.frames = [-1] + list(range(self.cr.first_frame, self.cr.last_frame+1))
    self.curr_frame = 0
    self.file = None

  def get_next(self):
    if not self.is_clip:
      if self.file and self.file.tell() < self.fsize:
        return self.file
      else:
        return None

    if self.file: self.file.close()

    while self.curr_frame < len(self.frames):
      frame = self.frames[self.curr_frame]
      self.file = self.file_from_buff(frame)
      self.curr_frame += 1
      if self.file:
        # looks like a classic generator... yield here you lazy bastard...
        return self.file

    return None

  def file_from_buff(self, frame):
    ret = self.cr.frame_info_ex(frame)
    if ret != 0: return None
    for fid in range(0, self.cr.files_num):
        name = self.cr.files[fid].name
        if name == GSF_PROF_RAW:
          size = self.cr.files[fid].size
          assert 0 == self.cr.read_file(frame, name, size), "error reading file %d, %d bytes from clipreader" % (fid, size)
          return io.BytesIO(self.cr.output_buffer)
    return None



class LazyRecords:
  def __init__(self, packet_bytes, curr_app, magic2record, version, dynamic_tasks, dependencies, time_scale, header=None):
    self.app = curr_app
    self._records = None
    self.packet_bytes = packet_bytes
    self.magic2record = magic2record
    self.min_record = min([r[1] for r in self.magic2record.values() if r])
    self.version = version
    self.dynamic_tasks = dynamic_tasks
    self.dependencies = dependencies
    self._tree = None
    self.header = header
    self.time_scale = time_scale

  @property
  def records(self):
    return list(self)

  @property
  def raw_records(self):
    return self._records

  def decode(self):
    valid_records = self.magic2record.keys()
    packet_size = len(self.packet_bytes)
    self._records = []

    pos = 0
    while pos < packet_size:
      assert pos + self.min_record <= packet_size, 'truncated file'

      record_magic = self.packet_bytes[pos]
      if type(record_magic) == str: record_magic = ord(record_magic)
      assert record_magic in valid_records, 'unknown record type "%c", valid records are:[%s]' % (record_magic, ','.join([chr(k) for k in valid_records]))

      record_type, record_size = self.magic2record[record_magic]
      assert pos + record_size <= packet_size, 'truncated file'

      self._records.append(record_type.from_buffer_copy(self.packet_bytes[pos:pos+record_size]))
      pos += record_size

    self.packet_bytes = None
    return self._records

  def __iter__(self):
    if not self._records:
      self.decode()
    # for r in self.records:
    #   yield r
    tree = self.tree()
    for goal in tree:
      yield goal
      for task in goal.tasks:
        yield task
        for job in task.jobs:
          yield job

  def tree(self, validate = True):
    if self._tree is not None:
      return self._tree
    if not self._records:
      self.decode()
    result = []
    # Data in these dicts is shared with the tree
    goal_tasks = defaultdict(list)
    goal_task_jobs = defaultdict(list)
    self._goals = {}
    self._tasks = {}
    self._goal_dyn_tasks = defaultdict(set)
    goals = self.app['agenda']['goals']
    last_task = None
    for r in self._records:
      if r.nesting_level == 0:
        r.tasks = goal_tasks[r.index]
        if 'g' in goals[r.index] and r._type is not None:
          assert r._type == goals[r.index]['g']
        elif r._type is None:
          r._type = goals[r.index]['g']
        r.name = goals[r.index]['n']
        result.append(r)
        self._goals[r.index] = r
      elif r.nesting_level == 1:
        if r.gindex is None:
          assert len(result) > 0, 'tasks without gindex must not precede their goal'
          r.gindex = result[-1].index
        r._type = self.get_task_type(r)
        r.jobs = goal_task_jobs[(r.gindex, r.index)] if r._type != GSF_WAIT_TASK else []
        if r.index >= 0:
          r.name = goals[r.gindex]['t'][r.index]
          r.fullname = r.name = goals[r.gindex]['t'][r.index]
          r.is_dynamic = False
        else:
          r.name = self.dynamic_tasks[r.tid].name
          r.fullname = None
          r.is_dynamic = True
        goal_tasks[r.gindex].append(r)
        self._tasks[(r.gindex, r.index)] = r
        last_task = r
      elif r.nesting_level == 2:
        if r.gindex is None:
          r.gindex = result[-1].index
        if r.tindex is None:
          r.tindex = last_task.index
          assert r.gindex == last_task.gindex
        if not hasattr(r, 'index'):
          r.index = len(goal_task_jobs[(r.gindex, r.tindex)])
        if type(r) is dep_prof_record:
          assert goal_tasks[r.gindex][-1].tindex == r.tindex
          goal_tasks[r.gindex][-1].dep = self.dependencies[r.dep_index]
        else:
          goal_task_jobs[(r.gindex, r.tindex)].append(r)
      else:
        raise RuntimeError("Unknown record type")
    for goal in result:
      goal.tasks.sort(key=lambda task: task.start)
      names = defaultdict(set)
      for task in goal.tasks:
        names[task.name].add(task.index)
      dupnames = {name: sorted(idx, reverse=True) for name, idx in names.items() if len(idx) > 1}
      for task in goal.tasks:
        if task.fullname is None:
          if task.name in dupnames:
            suffix = '.inst{}'.format(dupnames[task.name].index(task.index))
          else:
            suffix = ''
          if task.type == GSF_WAIT_TASK:
            suffix += '.wait'
          task.fullname = '{}.{}{}'.format(goal.name, task.name, suffix)
    self._tree = result
    if validate: self.validate_tree()
    return result

  def validate_tree(self):
    tree = self._tree
    assert len(tree) == len({g.index for g in tree}), 'Goals should be unique'
    goals = self.app['agenda']['goals']
    for goal in tree:
      not_dynamic_wait = [t.index for t in goal.tasks if t.index >= 0 or t.type == GSF_WAIT_TASK]
      assert len(not_dynamic_wait) == len(set(not_dynamic_wait)), \
        'Tasks should be unique except for dynamic wait tasks'
      assert self.header.start is None or goal.start >= self.header.start or goal.type == GSF_MSG_GOAL
      assert self.header.finish is None or goal.finish <= self.header.finish
      vmp = {}
      goal_descr = goals[goal.index]
      if 'g' in goal_descr and goal.type is not None:
        assert goal_descr['g'] == goal.type
      for t in goal.tasks:
        assert t.start >= goal.start and (t.finish <= goal.finish or goal.finish == 0)
        assert t.gindex == goal.index
        if t.index < 0 and t.type == GSF_VMP_PROC_TASK:
          vmp[t.index] = t
        if 'i' in goal_descr and t.type is not None and t.index >= 0 and goal_descr['i'][t.index] != GSF_OCL_KERNEL_TASK:
          assert t.type == goal_descr['i'][t.index]
        elif t.index < 0 and t.type == GSF_WAIT_TASK:
          assert t.index in vmp, 'WAIT cannot occur before VMP_PROC'
          assert vmp[t.index].finish <= t.finish
          assert vmp[t.index].start <= t.start
          del vmp[t.index]
        for job in t.jobs:
          assert job.start >= t.start # and (t.type != GSF_FOR_TASK or job.finish <= t.finish)
          assert job.tindex == t.index
          assert job.gindex == goal.index
          if job.magic == 'A': assert t.type == GSF_VMP_PROC_TASK
          if job.magic == 'F': assert t.type == GSF_FOR_TASK
      for i in range(1, len(goal.tasks)):
        assert goal.tasks[i-1].finish <= goal.tasks[i].start, \
          'The previous task must finish before the next one starts'

  def print_tree(self, file=sys.stdout):
    tree = self.tree()
    for goal in tree:
      print('{} {} {!s}'.format(
        goal.name, goal_type_names[goal.type], goal), file=file)
      for task in goal.tasks:
        print('  {} {} {}'.format(task.name, task_type_names[task.type], task), file=file)
        if task.dep is not None:
          print('    - D {}'.format(task.dep))
        for job in task.jobs:
          print('    {}'.format(job), file=file)

  def print_flat(self, data, file=sys.stdout):
    goals = self.app['agenda']['goals']
    for r in data:
      if r.nesting_level == 0:
        goal_descr = goals[r.index]
        rtype = goal_descr['g'] if 'g' in goal_descr else r.type
        print('{} {} {!s}'.format(
          goal_descr['n'], goal_type_names[rtype], r), file=file)
      elif r.nesting_level == 1:
        goal_descr = goals[r.gindex] if r.gindex is not None else {'t': {r.index: '???'}, 'i': {r.index: r.type}}
        if r.index >= 0:
          print('  {} {} {!s}'.format(
            goal_descr['t'][r.index], task_type_names[goal_descr['i'][r.index]] if r.type is not None else '', r)
          , file=file)
        else:
          print('  {} {} {}'.format(self.dynamic_tasks[r.tid].name, task_type_names[r.type] if r.type is not None else '', r), file=file)
      else:
        print('    {}'.format(r), file=file)

  def get_task_type(self, r):
    if r.index >= 0:
      if r.type is None:
        goal = self.app['agenda']['goals'][r.gindex]
        rtype = goal['i'][r.index]
      else:
        rtype = r.type
      if rtype == GSF_OCL_KERNEL_TASK:
        return GSF_VMP_PROC_TASK
      return rtype
    else:
      t = self.dynamic_tasks[r.tid].type
      dtasks = self._goal_dyn_tasks[r.gindex]
      if (t == GSF_FOR_TASK):
        return t
      else:
        if r.index in dtasks: # assume vmp/wait order was sorted correctly
          dtasks.remove(r.index)
          r.jobs = [] # FIXME: ugly hack
          return GSF_WAIT_TASK
        else:
          dtasks.add(r.index)
          return GSF_VMP_PROC_TASK

  def get_module(self, record):
    return self.app['agenda']['modules'][self.app['agenda']['goals'][record.gindex]['m']]

  def gic2ns(self, gic):
    return gic / self.time_scale

  def remove_agenda_prefix(self, name):
    prefix = self.app['agenda']['name'] + "."
    if name.startswith(prefix):
        return name[len(prefix):]
    else:
        return name

  def make_time_ranges(self, record, prev_line = None, prefix = None, priority = None):
    if hasattr(record, 'node'):
      assert type(record.node) is Node
      return record.node
    if record.nesting_level == 0:
      name = self.remove_agenda_prefix(record.name)
      priority = self.app['agenda']['goals'][record.index]['r']
    elif record.nesting_level == 1:
      name = self.remove_agenda_prefix(record.fullname)
    else:
      name = prefix \
        + (".j" if record.magic != ord('F') else '_g') + str(record.index)
    tr = TimeRange(start=self.gic2ns(record.start),
                   end=self.gic2ns(record.finish),
                   name=name,
                   line=(('MIPS', record.cid) if hasattr(record, 'cid') \
                         else (ACC_FAMILY[record.family], record.iid)),
                   module=self.get_module(record),
                   pre=[], dep=[], type=record.type, record=record, priority=priority)
    children = []
    if getattr(record, 'suspend', 0) > record.start:
      # A task that has two ranges
      record.ranges = [
        tr._replace(end=self.gic2ns(record.suspend), name=tr.name+" [suspend]", dep=[], pre=[], line=prev_line),
        tr._replace(start=self.gic2ns(record.resume), name=tr.name+" [resume]", dep=[], pre=[])
      ]
      depend(*record.ranges, DepKind.ON_END)
      if record.type == GSF_FOR_TASK and record.resume == 0:
          # todo: move to tree?
          assert record.suspend != 0
          record.resume = max(proc.finish for proc in record.jobs)
    elif getattr(record, 'tasks', []):
      # A goal that can have many ranges
      width = len(str(len(record.tasks)+1))
      record.ranges = [tr._replace(end=self.gic2ns(record.tasks[0].start), pre=[], dep=[],
                                   name=tr.name+' [{:0{}}]'.format(0, width))]
      vmp_tasks = {}
      vmp_names = {}
      frag_ends = [self.gic2ns(task.start) for task in record.tasks] + [tr.end]
      prev_line = tr.line
      for index, task in enumerate(record.tasks):
        children.append(self.make_time_ranges(task, prev_line, priority=priority))
        prev_line = children[-1].ranges[-1].line
        assert children[-1].end <= tr.end or tr.end == 0
        depend(record.ranges[-1], children[-1].ranges[0], DepKind.ON_END)
        if task.type == GSF_VMP_PROC_TASK:
          vmp_tasks[task.index] = task
          if task.index >= 0: # Ugly hack
            vmp_names[extract_task_subname(record.name, task.name, 'proc')] = task
        elif task.type == GSF_WAIT_TASK:
          if task.index >= 0:
            jobs = vmp_names[extract_task_subname(record.name, task.name, 'wait')].jobs
          else:
            jobs = vmp_tasks[task.index].jobs
          depend([job.ranges for job in jobs], task.ranges[-1], DepKind.ON_END)
        record.ranges.append(tr._replace(start=children[-1].ranges[-1].end,
                                         end=frag_ends[index+1],
                                         line=prev_line,dep=[],pre=[],
                                         name="{} [{:0{}}]".format(tr.name,index+1,width)))
        depend(children[-1].ranges[-1], record.ranges[-1], DepKind.ON_END)
        if task.type == GSF_FOR_TASK:
          depend([c.ranges for c in children[-1].children], record.ranges[-1], DepKind.ON_END)
    else:
      # A record with a single range
      record.ranges = [tr]
    if getattr(record, 'jobs', []):
      children = [self.make_time_ranges(job, prefix=name, priority=priority) for job in record.jobs]
      jobs_tr = [child.ranges for child in children]
      depend(record.ranges[0], jobs_tr, DepKind.ON_START)
      if record.dep:
        for dep, pres in enumerate(record.dep):
          depend([jobs_tr[pre] for pre in pres], jobs_tr[dep], DepKind.ON_END)
    record.node = Node(name, record.ranges, children, record, tr.start, tr.end)
    return record.node

  def time_range_tree(self):
    nodes = list(map(self.make_time_ranges, self.tree()))
    goals = self.app['agenda']['goals']
    for goal_idx, task_idx in self.app['agenda'].get('fake_deps', {}).items():
      goal = self._goals[int(goal_idx)]
      if tuple(task_idx) not in self._tasks:
        pre = self._goals[task_idx[0]].ranges[-1]
        # FIXME: this is wrong
        if pre.end <= goal.ranges[0].start:
          depend(pre, goal.ranges[0], DepKind.ON_END)
        continue # task did not run
      task = self._tasks[tuple(task_idx)]
      depend(task.ranges[0], goal.ranges[0], DepKind.ON_START)
      if goal.type == GSF_SYNC_GOAL and len(task.ranges) > 1:
        depend(goal.ranges[-1], task.ranges[-1], DepKind.ON_END)
    for node in nodes:
      info = goals[node.record.gindex]
      for pre in info['p']:
        gpre = self._goals[pre]
        if gpre.ranges:
          depend(gpre.ranges[-1], node.ranges[0], DepKind.ON_END)
        else:
          task_pre = self._tasks[tuple(self.app['agenda']['fake_deps'][str(pre)])]
          depend(task_pre.ranges[0], node.ranges[0], DepKind.ON_START)
    return nodes



# GSFProfDecoder decodes the self contained gsf-profile.raw
# use only read() to get a dict of applications
# { 'agenda': a json describing the agenda graph (goals, tasks, dependencies),
#   'frames': a dict of key=frame#, value=list of events stored in *_prof_record}
# for more information see http://wiki.mobileye.com/mw/index.php/GSF/TBB_profiling
# for sample usage, see __main__()
#
class GSFProfDecoder:
  def __init__(self, path, verbose = False):
    self.verbose = verbose
    self.apps = {}
    self.name2app = {}
    self.mips_freq = 0
    self.time_scale = 0
    self.eyeq_rev = 5
    self.rev_str = ""
    self.magic2record = None
    self.dynamic_tasks = []
    self.dependencies = {}
    self.source = GSFProfSource(path);
    self.file = self.source.get_next()

  def load_struct(self, struct_type):
    header_size = sizeof(struct_type)
    header_bytes = self.file.read(header_size)
    assert len(header_bytes) == header_size, 'truncated file (could not parse %s, read %d of %d bytes)' % (struct_type.__class__, len(header_bytes), header_size)

    header = struct_type.from_buffer_copy(header_bytes)
    assert check_magic(header, struct_type), 'invalid file format, expecting "%s" but got "%s"' % (magic_name(struct_type), struct.pack('I', header.magic).decode('utf-8'))

    return header, header_size

  def read_header(self):
    if self.verbose: print('read_header')
    header, _ = self.load_struct(prof_header)

    (self.mips_freq, self.time_scale) = init_record_versions(header.version)

    self.magic2record = make_magic2record()
    self.version = header.version

  def read_app_section(self):
    if self.verbose: print('read_app_section')

    header, header_size = self.load_struct(prof_app_section)
    section_size = header.size - header_size
    section_bytes = self.file.read(section_size)
    assert len(section_bytes) == section_size, 'truncated file (tried to read %d but read %d bytes)' % (section_size, len(section_bytes))

    if header.magic == prof_app_section.COMPRESSED_MAGIC:
      section_bytes = bz2.decompress(section_bytes)
    agenda = json.loads(section_bytes.decode("utf-8"))
    #hash_json(agenda)
    if header.app_id in self.apps:
      assert self.apps[header.app_id]['agenda'] == agenda, \
        'invalid format - multiple app_id 0x%x with different agenda' % header.app_id
      if self.verbose: print('duplicata agenda id: 0x%x' % (header.app_id))
      return
    if self.verbose: print('found agenda: "%s" id: 0x%x' % (agenda['name'], header.app_id))

    app = {'agenda': agenda, 'frames': {} }
    self.name2app[agenda['name']] = app
    self.apps[header.app_id] = app

  def read_cfg_section(self):
    if self.verbose: print('read_cfg_section')

    header, _ = self.load_struct(eyeq_cfg_section)
    self.pll = header.pll
    self.mips_freq = header.pll[PLL.CPU.value]
    self.time_scale = self.mips_freq * 1e-9
    self.eyeq_rev = header.eyeq_rev
    self.rev_str = header.rev_str
    if self.eyeq_rev == 6:
      global ACC_FAMILY
      ACC_FAMILY[3] = 'XNN'
      ACC_FAMILY[4] = 'XNN'

  def read_run_section(self):
    if self.verbose: print('read_run_section')

    # each header describes a whole "slow agenda" frame
    header, header_size = self.load_struct(prof_run_section)
    section_size = header.size - header_size
    packet_bytes = self.file.read(section_size)

    assert len(packet_bytes) == section_size, 'truncated file (tried to read %d but read %d bytes)' % (section_size, packet_size)
    assert header.app_id in self.apps, 'invalid format - missing app info for app_id 0x%x' % header.app_id
    if header.frame in self.apps[header.app_id]['frames']:
      print('WARNING: frame %d for app 0x%x already read - skipping'%(header.frame, header.app_id))
      return
    assert header.frame not in self.apps[header.app_id]['frames'], 'invalid format - duplicate frame records for app_id 0x%x frame #%d' % (header.app_id, header.frame)

    curr_app = self.apps[header.app_id]
    if self.verbose: print('adding events to frame %d in "%s" (%d bytes)' % (header.frame, self.apps[header.app_id]['agenda']['name'], header.size))
    curr_app['frames'][header.frame] = LazyRecords(packet_bytes, curr_app, self.magic2record, self.version, self.dynamic_tasks, self.dependencies, time_scale=self.time_scale, header=header)

  def read_dt_section(self):
    if self.verbose: print('read_dt_section')

    # dynamic tasks list
    header, header_size = self.load_struct(prof_dt_section)
    section_size = header.size - header_size
    packet_bytes = self.file.read(section_size)

    assert len(packet_bytes) == section_size, 'truncated file (tried to read %d but read %d bytes)' % (section_size, packet_size)
    dtasks = json.loads(packet_bytes.decode())
    self.dynamic_tasks += ( dynamic_task_info(type=t['t'],name=t['n']) for t in dtasks )

  def read_dep_section(self):
    if self.verbose: print('read_dep_section')

    # dynamic tasks list
    header, header_size = self.load_struct(prof_dep_section)
    section_size = header.size - header_size
    packet_bytes = self.file.read(section_size)

    assert len(packet_bytes) == section_size, 'truncated file (tried to read %d but read %d bytes)' % (section_size, packet_size)
    base_fmt = 'H'
    base_size = struct.calcsize(base_fmt)
    pos = 0
    table = []
    while pos < len(packet_bytes):
      l = struct.unpack_from(base_fmt, packet_bytes, pos)[0]
      fmt = '{:d}{}'.format(l, base_fmt)
      pos += base_size
      table.append(struct.unpack_from(fmt, packet_bytes, pos))
      pos += struct.calcsize(fmt)
    assert pos == len(packet_bytes)
    self.dependencies[header.index] = table

  def read(self):
    self.read_header()

    magic2section = { struct.pack('I', prof_app_section.COMPRESSED_MAGIC): self.read_app_section,
                      struct.pack('I', prof_app_section.UNCOMPRESSED_MAGIC): self.read_app_section,
                      struct.pack('I', eyeq_cfg_section.MAGIC): self.read_cfg_section,
                      struct.pack('I', prof_dt_section.MAGIC): self.read_dt_section,
                      struct.pack('I', prof_dep_section.MAGIC): self.read_dep_section,
                      struct.pack('I', prof_run_section_v7.MAGIC): self.read_run_section}
    valid_sections = magic2section.keys()
    while True:
      # expecting app/run sections or EOF
      magic_size = 4
      section = self.file.read(magic_size)
      if section == b'GSFp':
        print('WARNING: reading the file header (GSFp) in the middle of gsf-profile.raw')
        self.file.read(4)
        continue
      if len(section) == 0: # probably EOF
        self.file = self.source.get_next() # try skipping frame in clipreader
        if self.file:
          continue # will read magic again on the next file
        else:
          break

      assert len(section) == magic_size,  'truncated file - could not section header magic'
      assert section in valid_sections, 'unknown section type "%s", valid sections are:[%s]' % (section, ','.join([k.decode('utf-8') for k in valid_sections]))
      self.file.seek(-magic_size, 1) # rewind a bit so section readers will read an entire struct, 1=SEEK_SET

      magic2section[section]()
    return self.apps

if __name__ == '__main__':
  import argparse

  def sample_app_deps(app):
    modules = app['modules']
    goals = app['goals']
    for g in goals:
      print(modules[g['m']], g['n'], ' '.join([goals[x]['n'] for x in g['p']]))

  parser = argparse.ArgumentParser(description='GSF profile decoder')
  parser.add_argument('-p', '--path', help='Path to gsf-profile.raw or a clip', required=False, default=GSF_PROF_RAW)
  parser.add_argument('-d', '--demangle', help='Show goal/task name', required=False, action='store_true')
  parser.add_argument('-n', '--nanosec', help='Convert GIC to ns', required=False, action='store_true')
  parser.add_argument('-z', '--zero', help='Set time baseline (requires -n)', required=False, type=int, default=0)
  parser.add_argument('-v', '--verbose', help='Dump full summary', required=False, action='store_true')
  args = parser.parse_args()

  prof = GSFProfDecoder(args.path, verbose=args.verbose)
  apps = prof.read()

  print("EYEQ_REV = %d %s" % (prof.eyeq_rev, prof.rev_str));

  if hasattr(prof, 'pll'):
    for i in range(PLL.COUNT.value): # could iterate over enum, excluding "count"
      pll = PLL(i)
      print("%s freq = %.08fGHz" % (pll.name, prof.pll[pll.value] * 1e-9))

  for app in apps.values():
    name = app['agenda']['name']
    with open('gsf_app_%s.json'%name, 'wt') as f:
      json.dump(app['agenda'], f)
      #sample_app_deps(app['agenda'])

    goals = app['agenda']['goals']

    def gic2ns(ts):
      return int(ts / (prof.pll[PLL.CPU.value]*1e-9))

    def convert_ts(r):
      if args.nanosec:
        for f in ['start', 'finish', 'suspend', 'resume']:
          t = getattr(r, f, None)
          if t is not None: # don't care about 0
            setattr(r, f, gic2ns(t) - args.zero)

    with open('dynamic_tasks', 'wt') as f:
      f.write('\n'.join(map(str, prof.dynamic_tasks)))

    for frame, records in app['frames'].items():
      with open('gsf.%06d.prof'%frame, 'wt') as f:
        curr_goal = None
        for r in records:
          if args.nanosec and hasattr(prof, 'pll'):
            convert_ts(r)
          label = ''
          if args.demangle:
            if isinstance(r, goal_prof_record):
              curr_goal = r
              label = ' # ' + goals[r.index]['n']
            elif isinstance(r, task_prof_record):
              if r.index >= 0:
                label = ' # ' + goals[curr_goal.index]['t'][r.index]
              else:
                dt = prof.dynamic_tasks[r.tid];
                label = ' # ' + goals[curr_goal.index]['n'] + '_' + dt.name + ' ' + dynamic_task_info.t2s[r.type]
          print(str(r) + label, file=f)
