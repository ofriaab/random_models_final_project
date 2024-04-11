#!/usr/bin/python3
import gsf_prof
import sys


def main():
    prof = gsf_prof.GSFProfDecoder(sys.argv[1])
    apps = prof.read()
    for app in apps.values():
        for frame, records in app['frames'].items():
            with open('gsf.%06d.prof' % frame, 'wt') as f:
                def traverse(nodes):
                    for node in nodes:
                        for r in node.ranges:
                            print(r.name, r.line, r.start, r.end, r.priority, [d.name for d in r.dep], file=f)
                        traverse(node.children)

                traverse(records.time_range_tree())


main()
