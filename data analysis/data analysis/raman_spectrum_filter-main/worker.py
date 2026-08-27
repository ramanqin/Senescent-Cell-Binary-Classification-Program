import threading
import traceback
                                                                                           #从相应的模块导入
from state import Parameter_State
from core import Back_Filter_Excute

def Back_Worker ( params , log_function , progress_function  ):

    """

    The function is for worker.py to start core.py, the back filter main part.
    
    Args:

        params: dataclass of important args for back-end program to filter
        log_function : to print log on feedback tk.Text, no covering display
        progress_function : to show the progress of back-end, always covering display

    """

    log_function ( '\n====后端线程启动====')
    Back_Filter_Excute( params , log_function , progress_function)


def Worker_Start ( params , log_function , progress_function ):

    """
    
    The function is for worker.py to start core.py, the back filter main part.
    
    Args:

        params: dataclass of important args for back-end program to filter
        log_function : to print log on feedback tk.Text, no covering display
        progress_function : to show the progress of back-end, always covering display

    """

    Worker = threading.Thread(
                            target = Back_Worker ,
                            args = ( params , log_function , progress_function ) ,
                            daemon = True
                             )
    
    Worker.start()

    return Worker
