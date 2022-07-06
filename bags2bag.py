#!/usr/bin/env python

import roslib
import rosbag
import rospy
import sys
import argparse


def main(argv):

    inputfile1 = ''
    topics1 = ''
    inputfile2 = ''
    topics2 = ''
    outputfile = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='outputfile', required=True)
    parser.add_argument('-i1', dest='inputfile1', required=True)
    parser.add_argument('-t1', dest='topics1', nargs='*', type=str)
    parser.add_argument('-i2', dest='inputfile2', required=True)
    parser.add_argument('-t2', dest='topics2', nargs='*', type=str)
    parsed_args = parser.parse_args()

    inputfile1 = parsed_args.inputfile1
    inputfile2 = parsed_args.inputfile2
    outputfile = parsed_args.outputfile
    topics1 = parsed_args.topics1
    topics2 = parsed_args.topics2

    print ('Input 1 file is "', inputfile1)
    print ('Input 1 topics are "', topics1)
    print ('Input 2 file is "', inputfile2)
    print ('Input 2 topics are "', topics2)
    print ('Output file is "', outputfile)

    rospy.init_node('bag_combiner')

    outbag = rosbag.Bag(outputfile, 'w')

    print ("[ --- combine bags --- ]")

    try:
        for topic, msg, t in rosbag.Bag(inputfile1).read_messages(topics=topics1):
            if topic == "/tf" or topic == "/tf_static":
                outbag.write(topic, msg, msg.transforms[0].header.stamp)
            else:
                outbag.write(topic, msg, msg.header.stamp)

    finally:
        print ("")
        print (": Finished iterating through bag 1.")
        print ("")
    try:
        for topic, msg, t in rosbag.Bag(inputfile2).read_messages(topics=topics2):
            if topic == "/tf" or topic == "/tf_static":
                outbag.write(topic, msg, msg.transforms[0].header.stamp)
            else:
                outbag.write(topic, msg, msg.header.stamp)

    finally:
        print ("")
        print (": Finished iterating through bag 2.")
        print ("")
        outbag.close()


if __name__ == "__main__":
    main(sys.argv[1:])
