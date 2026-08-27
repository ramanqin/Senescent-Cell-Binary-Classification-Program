import pandas as pd
from tkinter import filedialog
from pathlib import Path
import tkinter as tk

def Import_Folder_Choose():     #定义了一个名为选择导入文件夹的函数

    # region initialization

    root_window = tk.Tk()       #使用tkinter库创建一个窗口对象，将其变量赋值给root window
    root_window.withdraw()      #隐藏窗口，防止弹出一个空白弹窗

    # region folder choose

    importing_file_folder_path = ''   #导入文件夹选择

    root_window.attributes("-topmost", True)     #将窗口设置为“始终置顶”
    root_window.update()                         #强制刷新窗口上所有待处理事件

    importing_file_folder_select = filedialog.askdirectory(parent = root_window ,        #弹出一个文件夹选择对话框，用户可以选择一个文件夹  #parent...指定对话框的父窗口
                                                            title = "请选择导入文件夹",    #标题文字
                                                            mustexist = True              #输入的必须真实存在
                                                            )
    importing_file_folder_path = Path(rf'{importing_file_folder_select}')                 #将用户的选择路径转换为Path对象，并赋值给importing_file_folder_path

    return importing_file_folder_path                                                     #返回选择的文件夹路径

def Export_Folder_Choose():                         #定义了一个名为选择导出文件夹的函数
                        
    # region initialization

    root_window = tk.Tk()                                                    
    root_window.withdraw() 

    # region folder choose

    exporting_file_folder_path = ''

    root_window.attributes("-topmost", True)                                                              #这段代码与上述导入文件夹代码同理
    root_window.update()

    exporting_file_folder_select = filedialog.askdirectory(parent = root_window ,
                                                            title = "请选择导出文件夹",
                                                            mustexist = True
                                                            )
    exporting_file_folder_path = Path(rf'{exporting_file_folder_select}')

    return exporting_file_folder_path

# def From_Folder_Read_Fullpath_Collection( Import_File_Folder_Path , File_Types = '*.txt'):

#     """
#     Function used to read file from certain folder, default (now only) full reading .txt

#     Args:
#         File_Types : list or single str, showing what kind of file should be read , as '*.txt'

#     """

#     Importing_Txt_Files = Import_File_Folder_Path.rglob ( File_Types )
#     Importing_Dfs = pd.read_csv ()

