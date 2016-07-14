"""Helper python script to generate debug messages"""
import datetime

#Colors for debug output
class ConsoleColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


INFO = ConsoleColors.OKGREEN + "[INFO]"
FATAL_ERROR = ConsoleColors.FAIL + "[FATAL_ERROR]"
WARNING = ConsoleColors.WARNING + "[WARNING]"

def get_current_time():
    """
        Returns the current system time
        @rtype: string
        @return: The current time, formatted
    """
    return str(datetime.datetime.now().strftime("%H:%M:%S"))

def print_status(type, message):
    """
        Prints a log message to the console
    """
    debug_status = type + " " + get_current_time() + " " + message + ConsoleColors.ENDC
    print(debug_status)