# encoding=utf-8
import sys
import threading
import time

class OutputLogger(object):
    """
    Manage the redirection of a copy of lines printed to stdout/stderr to a file
    
    Note: this will not redirect strings printed to stdout/stderr by sub-processes or native code extensions
    
    The updates are flushed to the file every 5 minutes or if 10Mb has been buffered
    
    Example Usage:
    
    # open the logger to write to a particular file
    logger = OutputLogger.open(os.path.join(os.environ["RESULT_DIR"],"log.txt"))
    
    # perform DL training, writing to stdout or stderr etc.  The logger will duplicate printed output to the specified path
    # output will be automatically sync'd to file at least every 5 minutes but a sync can be forced by calling logger.syncBuffer()
    
    # close the logger once the activity is complete
    logger.close()
    """

    def __init__(self,filename,buffer_size,sync_interval):
        self.filename = filename

        self.buffer = []
        self.buffer_sz = 0

        self.buffer_sz_threshold = buffer_size

        self.orig_stdout = sys.stdout

        sys.stderr = OutputLogger.Redirect(self,sys.stderr)
        sys.stdout = OutputLogger.Redirect(self,sys.stdout)

        self.lock = threading.Lock()

        self.last_sync = time.time()
        self.sync_interval = sync_interval

        def run():
            self.__monitor()

        self.running = True
        self.monitorThread = threading.Thread(target=run)
        self.monitorThread.start()

    class Redirect(object):
        """
        Utility class to wrap stdout or stderr streams, providing write and flush methods
        """

        def __init__(self, logger, orig_stream):
            self.logger = logger
            self.orig_stream = orig_stream

        def write(self, m):
            self.logger.appendBuffer(m)
            self.orig_stream.write(m)

        def flush(self):
            self.orig_stream.flush()

        def getOriginalStream(self):
            return self.orig_stream

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

    @staticmethod
    def open(filename,buffer_size=10*1024*1024,sync_interval=300):
        """
        Open an OutputLogger backed by a file and start duplicating python's stdout/stderr to it
        
        All updates are appended to the file, any existing content will not be removed
        
        :param filename: path to the file 
        :param buffer_size: optional - number of characters to buffer before flushing to file (default 10Mb)
        :param sync_interval: optional - interval in seconds between file syncs (default 5 minutes)
        :return: an OutputLogger object
        """
        logger = OutputLogger(filename,buffer_size,sync_interval)
        logger.syncBuffer()
        return logger

    def close(self):
        """
        Close the OutputLogger, flushing any buffered output, and disable duplication
        """
        self.syncBuffer()
        sys.stderr = sys.stderr.getOriginalStream()
        sys.stdout = sys.stdout.getOriginalStream()
        self.running = False
        self.monitorThread.join()

    def __monitor(self):
        # periodically check to see if the OutputLogger is still running and
        # sync the file if enough time has elapsed since the last sync
        while self.running:
            time.sleep(10)
            now = time.time()
            if now - self.last_sync > self.sync_interval:
                self.syncBuffer()

    def appendBuffer(self,m):
        # append a line to the OutputLogger's buffer

        if not self.running:
            return

        self.lock.acquire()
        try:
            self.buffer.append(m)
            self.buffer_sz += len(m)
        finally:
            self.lock.release()
        # flush the log if the buffer has become too full
        if self.buffer_sz >= self.buffer_sz_threshold:
            self.syncBuffer()

    def syncBuffer(self):
        """
        Manually sync the buffered contents to file
        """
        self.lock.acquire()
        try:
            # do this even if there is nothing buffered so that the file will at least get created
            with open(self.filename, "a") as buffer_file:
                for m in self.buffer:
                    buffer_file.write(m)
                self.buffer = []
                self.buffer_sz = 0
            self.last_sync = time.time()
        finally:
            self.lock.release()

