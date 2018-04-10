from output_logger import OutputLogger
import time
import sys

with OutputLogger.open("/tmp/ol.txt",1024,30) as ol:
    # /tmp/ol.txt should be created but empty at this point
    print("00001. Printing something")
    sys.stderr.write("00002. Write something to stderr\n")
    # /tmp/ol.txt still empty
    print("00003. Calling syncBuffer")
    ol.syncBuffer()
    # ol.txt should contain 3 lines now
    for index in range(4,150):
        print("%05d. <>"%(index))
    # ol.txt should contain ~105 lines now
    print("00150. sleeping 60 seconds before exit")
    time.sleep(60)
    # ol.txt should contain all 150 lines now

print("This should not be duplicated to ol.txt")

