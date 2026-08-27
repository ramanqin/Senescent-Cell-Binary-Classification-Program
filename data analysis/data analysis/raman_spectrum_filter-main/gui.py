import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
import shutil
import bisect
import json
from dataclasses import dataclass, asdict

import tkinter as tk 
from tkinter import ttk
from tkinter import filedialog
from tkinter import font 

import threading
import queue
import traceback

from state import Parameter_State
from worker import Worker_Start

def App():

    content_height = 84
    content_sticky = 'nswe'

    default_params = Parameter_State()

    Log_Queue = queue.Queue()
    Progress_Log_Queue = queue.Queue()

    # region root_window

    root_window = tk.Tk()
    root_window.geometry ('1440x960')
    root_window.configure ( bg = '#AECEEE')

    root_window.after(100, lambda: root_window.attributes("-topmost", True))
    root_window.after(150, root_window.lift)
    root_window.after(150, root_window.focus_force)
    root_window.after(100, lambda: root_window.attributes("-topmost", False))

    root_window.rowconfigure( 1 , weight = 2 )
    root_window.rowconfigure( 2 , weight = 2 )
    root_window.rowconfigure( 3 , weight = 5 )
    root_window.rowconfigure( 4 , weight = 2 )
    root_window.rowconfigure( 5 , weight = 5 )
    root_window.rowconfigure( 6 , weight = 5 )

    root_window.columnconfigure ( 0 , weight= 1)
    root_window.columnconfigure ( 1 , weight= 2)

    Peak_Height_Ratio = tk.DoubleVar(master= root_window)

    # endregion

    # region style

    '''
    OFL_Noto Sans SC
    OFL_Noto Sans SC Black
    OFL_Noto Sans SC ExtraBold
    OFL_Noto Sans SC ExtraLight
    OFL_Noto Sans SC Light
    OFL_Noto Sans SC Medium
    OFL_Noto Sans SC SemiBold
    OFL_Noto Sans SC Thin

    OFL_Noto Sans
    OFL_Noto Sans Black
    OFL_Noto Sans ExtraBold
    OFL_Noto Sans ExtraLight
    OFL_Noto Sans Light
    OFL_Noto Sans Medium
    '''
    main_title_style = ttk.Style()
    main_title_style.configure(
        'main_title.TLabel',
        font = ( 'OFL_Noto Sans SC Medium' , 36  ),
        padding = 0
    )

    version_title_style = ttk.Style()
    version_title_style.configure(
        'version_title.TLabel',
        font = ( 'OFL_Noto Sans Medium' , 12 ) ,
        padding = 0
    )

    option_frame_style = ttk.Style() 
    option_frame_style.configure(
        'option_frame.TFrame',
        # background = '#C0C0C0' , 
        padding = 8
    )

    content_label_style = ttk.Style()
    content_label_style.configure(
        'content_label.TLabel' , 
        font = ( 'OFL_Noto Sans SC'  , 12) , 
        foreground = '#000000' ,
        padding = 0
    )

    subtitle_label_style = ttk.Style()
    subtitle_label_style.configure(
        'subtitle_label.TLabel' , 
        font = ( 'OFL_Noto Sans SC Black'  , 20) , 
        foreground = '#000000' ,
        padding = 0
    )

    # endregion

    # region title

    title_frame = ttk.Frame (root_window ,
                            #  relief="solid", 
                            #  borderwidth= 0 , 
                            height= 120
                            )
    title_frame.grid( row = 0 ,column = 0  ,  columnspan= 2 , sticky= 'ew' ,pady =(0 , 10))
    title_frame.columnconfigure ( 0 , weight= 1 )
    title_frame.grid_propagate (True)

    title_underline =  tk.Frame (title_frame , height = 2 , bg = '#101010')
    title_underline.grid ( row = 2 , column = 0 , sticky= 'sew')

    main_title = ttk.Label (title_frame , text = '大肠杆菌拉曼光谱清洗程序' , style='main_title.TLabel')
    main_title.grid(row = 0 , column = 0  ,padx = 0 , pady = 0 )
    
    version_title = ttk.Label (title_frame , text= 'ver 1.6' , style = 'version_title.TLabel')
    version_title.grid ( row = 1 , column = 0  , sticky='n')

    # endregion

    class Band_Selector(ttk.Frame):

        def __init__(self , parent , subtitle , width = 256, height = 64 , padx = 6, pady = 6):

            super().__init__(parent)
            self.Start_Var = tk.DoubleVar()
            self.End_Var = tk.DoubleVar()

            self.grid_propagate (False)
            self.configure ( width= width , height= height)
            self.grid ( padx = padx , pady = pady)
            self.configure ( padding = 6 )

            self.columnconfigure ( 0 , weight= 3)
            self.columnconfigure ( 1 , weight= 1)
            self.columnconfigure ( 2 , weight= 3)
            self.columnconfigure ( 3 , weight= 1)
            self.rowconfigure ( 0 , weight= 1)
            self.rowconfigure ( 1 , weight= 1)

            self.label_name = ttk.Label(
                                    self , 
                                    text= subtitle
            )
            self.label_name.grid( row = 0 ,column = 0 , columnspan= 2 ,pady= ( 0 , 10 ))

            self.label_up = ttk.Label (
                                    self , 
                                    text= '区域范围'
            )
            self.label_up.grid( row = 0 ,column = 2 , columnspan= 2 ,pady= ( 0 ,10 ))

            self.label_1 = ttk.Label(
                                    self , 
                                    text= '起始：' ,
            )
            self.label_1.grid ( row = 1 , column = 0 )   

            self.entry_1 = ttk.Entry(
                                    self , 
                                    textvariable= self.Start_Var,  
                                    width= 7                           
            )
            self.entry_1.grid ( row= 1 ,column= 1)

            self.label_2 = ttk.Label(
                                    self , 
                                    text= '结束：'
            )
            self.label_2.grid ( row = 1 , column = 2)

            self.entry_2 = ttk.Entry(
                                    self , 
                                    textvariable= self.End_Var,    
                                    width = 7                          
            )
            self.entry_2.grid ( row= 1 ,column= 3)
        

        def text_style_apply( self , text_style):

            self.label_1.configure ( style= text_style)
            self.label_2.configure ( style= text_style)
            self.label_up.configure ( style= text_style)

        def title_style_apply ( self , title_style):

            self.label_name.configure ( style= title_style)

        def entry_font ( self , font_face , font_size):

            self.entry_1.configure(font = (font_face , font_size))
            self.entry_2.configure(font = (font_face , font_size))

    class SNR_Selector(ttk.Frame):

        def __init__(self , parent , width = 256, height = 64 , padx = 6, pady = 6):
            super().__init__(parent)
            self.Min_SNR_Var = tk.DoubleVar()
            self.Max_Standard_Error = tk.DoubleVar()

            self.grid_propagate(False)
            self.configure ( width= width , height= height , padding= 6)
            self.grid ( padx = padx , pady = pady)
            self.rowconfigure(0 , weight=1)
            self.rowconfigure(1 , weight=1)
            self.columnconfigure( 0 , weight= 2)
            self.columnconfigure ( 1 , weight= 1)

            self.label_1 = ttk.Label(
                                    self , 
                                    text = '信噪比最低标准：'
            )
            self.label_1.grid( row= 0 ,column= 0)

            self.entry_1 = ttk.Entry(
                                    self , 
                                    width= 8 ,
                                    textvariable= self.Min_SNR_Var
            )
            self.entry_1.grid ( row= 0 , column= 1)

            self.label_2 = ttk.Label(
                                    self , 
                                    text = '标准差最高标准：'
            )
            self.label_2.grid( row= 1 ,column= 0)

            self.entry_2 = ttk.Entry(
                                    self , 
                                    width= 8 ,
                                    textvariable= self.Max_Standard_Error
            )
            self.entry_2.grid ( row= 1 , column= 1)
            
        def text_style_apply( self , text_style):

            self.label_1.configure ( style= text_style)
            self.label_2.configure ( style= text_style)


        def entry_font ( self , font_face , font_size):

            self.entry_1.configure(font = (font_face , font_size))
            self.entry_2.configure(font = (font_face , font_size))

    class Peak_Selector(ttk.Frame):

        def __init__(self , parent , Peak_Height_Ratio , width = 256, height = 64 , padx = 6, pady = 6 ):
            super().__init__(parent)
            self.Min_Peak_Length = tk.DoubleVar()
            self.Min_SNR_Var = tk.DoubleVar()
            self.Min_Peak_Height = tk.DoubleVar()
            self.Peak_Max_Standard_Error = tk.DoubleVar()
            self.Noise_Max_Standard_Error = tk.DoubleVar()
            self.Peak_Height_Ratio = Peak_Height_Ratio

            self.grid_propagate(False)
            self.configure ( width= width , height= height , padding= 6)
            self.grid ( padx = padx , pady = pady)
            self.columnconfigure( 0 , weight= 1)
            self.columnconfigure ( 1 , weight= 1)
            self.rowconfigure( 0 , weight= 1)
            # self.rowconfigure( 1 , weight= 1)

            self.frame_left = ttk.Frame(self)
            self.frame_left.grid ( row= 0 , column= 0)
            self.frame_right = ttk.Frame(self)
            self.frame_right.grid (row=0 , column= 1)

            self.frame_left.columnconfigure ( 0 , weight= 2)
            self.frame_left.columnconfigure ( 1 , weight= 1)
            self.frame_left.rowconfigure ( 0 , weight= 1)
            self.frame_left.rowconfigure ( 1 , weight= 1)
            self.frame_left.rowconfigure ( 2 , weight= 1)


            self.frame_right.columnconfigure ( 0 , weight= 2)
            self.frame_right.columnconfigure ( 1 , weight= 1)
            self.frame_right.rowconfigure ( 0 , weight= 1)
            self.frame_right.rowconfigure ( 1 , weight= 1)
            self.frame_right.rowconfigure ( 2 , weight= 1 )
            
            self.label_1 = ttk.Label(
                                    self.frame_left , 
                                    text = '信噪比最低标准：'
            )
            self.label_1.grid( row= 0 ,column= 0)

            self.entry_1 = ttk.Entry(
                                    self.frame_left , 
                                    width= 8 ,
                                    textvariable= self.Min_SNR_Var
            )
            self.entry_1.grid ( row= 0 , column= 1)

            self.label_2 = ttk.Label(
                                    self.frame_left , 
                                    text = '最小峰长（/个采样点）：'
            )
            self.label_2.grid( row = 1  , column=  0)

            self.entry_2 = ttk.Entry(
                                    self.frame_left , 
                                    width= 8 ,
                                    textvariable= self.Min_Peak_Length
            )
            self.entry_2.grid ( row= 1 , column= 1)

            self.label_3 = ttk.Label(
                                    self.frame_left , 
                                    text = '最小峰高（/个标准差）：'
            )
            self.label_3.grid( row = 2  , column=  0)

            self.entry_3 = ttk.Entry(
                                    self.frame_left , 
                                    width= 8 ,
                                    textvariable= self.Min_Peak_Height
            )
            self.entry_3.grid ( row= 2 , column= 1)



            self.label_4 = ttk.Label(
                                    self.frame_right , 
                                    text = '峰区标准差最高：'
            )
            self.label_4.grid( row= 0 ,column= 0)

            self.entry_4 = ttk.Entry(
                                    self.frame_right , 
                                    width= 8 ,
                                    textvariable= self.Peak_Max_Standard_Error
            )
            self.entry_4.grid ( row= 0 , column= 1)

            self.label_5 = ttk.Label(
                                    self.frame_right , 
                                    text = '非峰区标准差最高：'
            )
            self.label_5.grid( row= 1 ,column= 0)

            self.entry_5 = ttk.Entry(
                                    self.frame_right , 
                                    width= 8 ,
                                    textvariable= self.Noise_Max_Standard_Error
            )
            self.entry_5.grid ( row= 1 , column= 1)

            self.label_6 = ttk.Label(
                                    self.frame_right , 
                                    text = '峰高比（CH/指纹）下限：'
            )
            self.label_6.grid( row= 2 ,column= 0)

            self.entry_6 = ttk.Entry(
                                    self.frame_right , 
                                    width= 8 ,
                                    textvariable= self.Peak_Height_Ratio
            )
            self.entry_6.grid ( row= 2 , column= 1)


        def text_style_apply( self , text_style):

            self.label_1.configure ( style= text_style)
            self.label_2.configure ( style= text_style)
            self.label_3.configure ( style= text_style)
            self.label_4.configure ( style= text_style)
            self.label_5.configure ( style= text_style)
            self.label_6.configure ( style= text_style)


        def entry_font ( self , font_face , font_size):

            self.entry_1.configure(font = (font_face , font_size))
            self.entry_2.configure(font = (font_face , font_size))
            self.entry_3.configure(font = (font_face , font_size))
            self.entry_4.configure(font = (font_face , font_size))
            self.entry_5.configure(font = (font_face , font_size))
            self.entry_6.configure(font = (font_face , font_size))

    class Folder_Selector(ttk.Frame):

        def __init__(self , parent , width = 128, height = 24 , padx = 12, pady = 6, text_showing = '预设：选择文件夹'):

            super().__init__(parent)

            self.Folder_Path = tk.StringVar()

            self.grid_propagate(False)
            self.configure ( width= width , height= height , padding= 6)
            self.grid ( padx = padx , pady = pady)
            self.rowconfigure(0 , weight=1)
            self.columnconfigure( 0 , weight= 2)
            self.columnconfigure( 1 , weight= 6)
            self.columnconfigure( 2 , weight= 2)
            
            self.label = ttk.Label(
                                    self , 
                                    text = text_showing
            )
            self.label.grid ( row = 0 , column= 0) 

            self.entry = ttk.Entry(
                                    self , 
                                    textvariable= self.Folder_Path
            )
            self.entry.grid ( row = 0 ,column= 1 , sticky= 'we')

            self.button = ttk.Button(
                                        self,
                                        text= '选择文件夹' ,
                                        command = lambda:self.Select_Folder(text_showing) ,
            )
            self.button.grid ( row = 0 ,column= 2 ,  sticky= 'we' , padx= (12 , 6))

        def text_style_apply( self , text_style):

            self.label.configure ( style= text_style)

        def entry_font ( self , font_face , font_size):

            self.entry.configure(font = (font_face , font_size))

        def Select_Folder (self , text_on_dialog):

            folder_select = filedialog.askdirectory(
                                                    parent = root_window ,
                                                    title = text_on_dialog,
                                                    mustexist = True
                                                    )
            folder_path = Path(rf'{folder_select}')
            self.Folder_Path.set (  folder_select )

    class Feedback(ttk.Frame):

        def __init__(self , parent , width = 128, height = 192 , padx = 12, pady = 6):

            super().__init__(parent)

            self.Print_Content = tk.StringVar()
            self.Status_Log = tk.StringVar ( value = '进度：' )

            self.grid_propagate(False)
            self.configure ( width= width , height= height , padding= 6)
            self.grid ( padx = padx , pady = pady)
            self.rowconfigure(0 , weight=0)
            self.rowconfigure(1 , weight=1)
            self.columnconfigure( 0 , weight= 1)
            self.columnconfigure( 1 , weight= 1)
            self.columnconfigure( 2 , weight= 1)


            
            self.label = ttk.Label(
                                    self , 
                                    text = '工作日志：'
            )
            self.label.grid ( row = 0 , column= 0 , sticky= 'w') 

            self.button = ttk.Button(
                                    self , 
                                    text= '开始筛选' , 
                                    command= lambda:Gui_Start_Button()
            )
            self.button.grid ( row = 0 , column= 2 , sticky= 'e') 

            self.status_label = ttk.Label(
                                    self , 
                                    textvariable= self.Status_Log
            )
            self.status_label.grid ( row = 0 , column= 1 ) 

            self.text = tk.Text(
                                self,
                                
            )
            self.text.grid( row = 1 , column = 0 , columnspan= 3 , sticky= 'ew' )

        def text_style_apply( self , text_style):

            self.label.configure ( style= text_style)
            self.status_label.configure ( style= text_style)

    # region front core

    input_folder = Folder_Selector(root_window ,text_showing= '选择导入文件夹' , height = 66)
    input_folder.grid( row= 1 , column= 0  , columnspan= 2 ,padx = 12 , sticky= content_sticky)
    input_folder.text_style_apply('content_label.TLabel')
    input_folder.entry_font( 'OFL_Noto Sans Medium' , 12)

    output_folder = Folder_Selector(root_window ,text_showing= '选择导出文件夹' , height = 66)
    output_folder.grid( row= 2 , column= 0  , columnspan= 2 ,padx = 12 , sticky= content_sticky)
    output_folder.text_style_apply('content_label.TLabel')
    output_folder.entry_font( 'OFL_Noto Sans Medium' , 12)



    finger_2 = Band_Selector(root_window , height= int(content_height*2) , subtitle= "指纹区筛选条件" )
    finger_2.grid ( row= 3 ,column= 0 , sticky= content_sticky)
    finger_2.grid (padx = (12,6))

    finger_2.title_style_apply ( 'subtitle_label.TLabel')
    finger_2.text_style_apply('content_label.TLabel')
    finger_2.entry_font( 'OFL_Noto Sans Medium' , 12)

    finger_3 = Peak_Selector(root_window , Peak_Height_Ratio= Peak_Height_Ratio , height= int(content_height*2) )
    finger_3.grid ( padx = ( 6,12 ))
    finger_3.grid ( row= 3 ,column= 1 , sticky= content_sticky)

    finger_3.text_style_apply('content_label.TLabel')
    finger_3.entry_font( 'OFL_Noto Sans Medium' , 12)



    silence_2 = Band_Selector(root_window , height=  int(content_height*3/2) ,subtitle='静默区筛选条件')
    silence_2.grid ( row= 4 ,column= 0 , sticky= content_sticky)
    silence_2.grid(padx = (12,6))

    silence_2.title_style_apply ( 'subtitle_label.TLabel')
    silence_2.text_style_apply('content_label.TLabel')
    silence_2.entry_font( 'OFL_Noto Sans Medium' , 12)

    silence_3 = SNR_Selector(root_window , height= int(content_height*3/2))
    silence_3.grid ( padx = ( 6,12 ))
    silence_3.grid ( row= 4 ,column= 1 , sticky= content_sticky)

    silence_3.text_style_apply('content_label.TLabel')
    silence_3.entry_font( 'OFL_Noto Sans Medium' , 12)





    C_H_2 = Band_Selector(root_window , height= int(content_height*2) , subtitle= "C-H峰筛选条件")
    C_H_2.grid ( row= 5 ,column= 0 , sticky= content_sticky)
    C_H_2.grid(padx = (12,6))

    C_H_2.title_style_apply ( 'subtitle_label.TLabel')
    C_H_2.text_style_apply('content_label.TLabel')
    C_H_2.entry_font( 'OFL_Noto Sans Medium' , 12)

    C_H_3 = Peak_Selector(root_window , Peak_Height_Ratio= Peak_Height_Ratio , height=  int(content_height*2) )
    C_H_3.grid ( padx = ( 6,12 ))
    C_H_3.grid ( row= 5 ,column= 1 , sticky= content_sticky)

    C_H_3.text_style_apply('content_label.TLabel')
    C_H_3.entry_font( 'OFL_Noto Sans Medium' , 12)



    Terminal = Feedback ( root_window , height = 256)
    Terminal.grid( row= 6 , column= 0 , columnspan= 2 , padx = 12 , sticky= content_sticky) 
    Terminal.text_style_apply('content_label.TLabel')

    # endregion

    def Parameter_Get ():

        Parameter_Gotten = Parameter_State(

                                            Input_Folder_Path = input_folder.Folder_Path.get() , 
                                            Output_Folder_Path = output_folder.Folder_Path.get() ,

                                            Finger_Peak_Min_Length = finger_3.Min_Peak_Length.get() ,  
                                            Finger_Peak_Min_Height = finger_3.Min_Peak_Height.get() , 
                                            Finger_Peak_Max_STD = finger_3.Peak_Max_Standard_Error.get() , 
                                            Finger_Noise_Max_STD = finger_3.Noise_Max_Standard_Error.get() , 
                                            Finger_Min_SNR = finger_3.Min_SNR_Var.get () ,

                                            Peak_Height_Ratio = Peak_Height_Ratio.get() , 

                                            Silence_Max_STD = silence_3.Max_Standard_Error.get() , 
                                            Silence_Min_SNR = silence_3.Min_SNR_Var.get() , 

                                            CH_Peak_Min_Length = C_H_3.Min_Peak_Length.get() , 
                                            CH_Peak_Min_Height = C_H_3.Min_Peak_Height.get() , 
                                            CH_Peak_Max_STD = C_H_3.Peak_Max_Standard_Error.get() , 
                                            CH_Noise_Max_STD = C_H_3.Noise_Max_Standard_Error.get () , 
                                            CH_Min_SNR = C_H_3.Min_SNR_Var.get ( ) , 

                                            Finger_Start = finger_2.Start_Var.get() , 
                                            Finger_End = finger_2.End_Var.get() , 
                                            Silence_Start = silence_2.Start_Var.get() , 
                                            Silence_End = silence_2.End_Var.get() , 
                                            CH_Start = C_H_2.Start_Var.get() , 
                                            CH_End = C_H_2.End_Var.get() 

                                            ) 
        return Parameter_Gotten
    
    def Parameter_Set ( params ):

        input_folder.Folder_Path.set(params.Input_Folder_Path)
        output_folder.Folder_Path.set(params.Output_Folder_Path)

        finger_3.Min_Peak_Length.set( params.Finger_Peak_Min_Length )
        finger_3.Min_Peak_Height.set( params.Finger_Peak_Min_Height )
        finger_3.Peak_Max_Standard_Error.set( params.Finger_Peak_Max_STD )
        finger_3.Noise_Max_Standard_Error.set( params.Finger_Noise_Max_STD )
        finger_3.Min_SNR_Var.set ( params.Finger_Min_SNR )

        silence_3.Max_Standard_Error.set (params.Silence_Max_STD ) 
        silence_3.Min_SNR_Var.set( params.Silence_Min_SNR ) 

        C_H_3.Min_Peak_Length.set(params.CH_Peak_Min_Length ) 
        C_H_3.Min_Peak_Height.set(params.CH_Peak_Min_Height) 
        C_H_3.Peak_Max_Standard_Error.set(params.CH_Peak_Max_STD ) 
        C_H_3.Noise_Max_Standard_Error.set (params.CH_Noise_Max_STD ) 
        C_H_3.Min_SNR_Var.set (params.CH_Min_SNR  ) 

        finger_2.Start_Var.set(params.Finger_Start ) 
        finger_2.End_Var.set(params.Finger_End ) 
        silence_2.Start_Var.set(params.Silence_Start ) 
        silence_2.End_Var.set(params.Silence_End ) 
        C_H_2.Start_Var.set(params.CH_Start ) 
        C_H_2.End_Var.set(params.CH_End ) 
        
        Peak_Height_Ratio.set ( params.Peak_Height_Ratio )

    def Poll_Log ( ):

        while not Log_Queue.empty():

            message_on_board = Log_Queue.get()

            Terminal.text.insert( "end" , message_on_board)
            Terminal.text.see ( 'end' )

        root_window.after(1000 , Poll_Log)

    def Poll_Progress():

        while not Progress_Log_Queue.empty():

            message = Progress_Log_Queue.get()    
            Terminal.Status_Log.set(message)

        root_window.after(200 , Poll_Progress)

    def Print_Log_Put(message):
        
        Log_Queue.put(message)

    def Progress_Log_Put(message):

        Progress_Log_Queue.put(message)

    def Gui_Start_Button():

        parmas = Parameter_Get()
        Worker_Start ( parmas, log_function = Print_Log_Put , progress_function = Progress_Log_Put)

    def Save_Config ( ):

        config_save_path =  filedialog.asksaveasfilename(
                                                    title="导出参数配置文件",
                                                    defaultextension=".json",
                                                    filetypes=[("JSON files", "*.json")]
                                                    )
        params = Parameter_Get()
    
        if not config_save_path:
            return

        with open(config_save_path, "w", encoding="utf-8") as f:

            json.dump(
                    asdict(params),
                    f,
                    ensure_ascii=False,
                    indent=4
                    )
            
    def Load_Config ():
        
        config_load_path = filedialog.askopenfilename(
                                                    title="加载参数配置文件",
                                                    filetypes=[("JSON files", "*.json")]
                                                    )

        if not config_load_path:
            return

        with open(config_load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        Parameter_Set ( Parameter_State (**data) )

    # region menu 

    menu_bar = tk.Menu(root_window)

    parameter_menu = tk.Menu ( menu_bar , tearoff= False)

    parameter_menu.add_command(
                            label="加载参数配置",
                            command = Load_Config
                            )

    parameter_menu.add_command(
                            label="导出参数配置",
                            command= Save_Config 
                            )

    menu_bar.add_cascade(label="参数设置", menu = parameter_menu)     
    root_window.config ( menu = menu_bar )

    # endregion

    Parameter_Set ( default_params )

    Poll_Progress()
    Poll_Log()

    root_window.mainloop()

    


    # def log(message):

    #     log_queue.put(message)

    # def status_log(message):

    #     update_log_queue.put(message)

    # def update_log():

    #     while not update_log_queue.empty():

    #         messge = update_log_queue.get()
    #         Terminal.Status_Log.set(messge)   # 直接覆盖上一条内容
    
    # def Check_Log_Queue ( ):

    #     while not log_queue.empty():

    #         message_get = log_queue.get()

    #         if message_get == '__DONE__':
                
    #             Terminal.button.configure(state="normal")
    #             Terminal.text.insert("end", "\n筛选任务结束。\n")
    #             Terminal.text.see("end")

    #         else:

    #             Terminal.text.insert("end", message_get)
    #             Terminal.text.see("end")  

    #     root_window.after(1000, Check_Log_Queue)
    #     root_window.after(1000, update_log)  

    # def Start_Execute_Thread():

    #     # region parameter

    #     try:
    #         params = {
    #             "input_folder_path": input_folder.Folder_Path.get(),
    #             "output_folder_path": output_folder.Folder_Path.get(),

    #             "start_1": finger_2.Start_Var.get(),
    #             "end_1": finger_2.End_Var.get(),

    #             "start_2": silence_2.Start_Var.get(),
    #             "end_2": silence_2.End_Var.get(),

    #             "start_3": C_H_2.Start_Var.get(),
    #             "end_3": C_H_2.End_Var.get(),

    #             "peak_height_1": finger_3.Min_Peak_Height.get(),
    #             "peak_length_1": finger_3.Min_Peak_Length.get(),
    #             "snr_min_1": finger_3.Min_SNR_Var.get(),
    #             "peak_std_max_1": finger_3.Peak_Max_Standard_Error.get() , 
    #             "noise_std_max_1": finger_3.Noise_Max_Standard_Error.get() , 

    #             "snr_min_2": silence_3.Min_SNR_Var.get(),
    #             "std_max_2": silence_3.Max_Standard_Error.get() ,

    #             "peak_height_3": C_H_3.Min_Peak_Height.get(),
    #             "peak_length_3": C_H_3.Min_Peak_Length.get(),
    #             "snr_min_3": C_H_3.Min_SNR_Var.get(),
    #             "peak_std_max_3": C_H_3.Peak_Max_Standard_Error.get() , 
    #             "noise_std_max_3": C_H_3.Noise_Max_Standard_Error.get() , 

    #             'peak_height_ratio':C_H_3.Peak_Height_Ratio.get()
    #         }

    #     except Exception:
    #         Terminal.text.insert("end", "\n参数读取失败：\n")
    #         Terminal.text.insert("end", traceback.format_exc())
    #         Terminal.text.see("end")
    #         return

    #     if not params["input_folder_path"]:
    #         Terminal.text.insert("end", "\n请先选择导入文件夹。\n")
    #         Terminal.text.see("end")
    #         return

    #     if not params["output_folder_path"]:
    #         Terminal.text.insert("end", "\n请先选择导出文件夹。\n")
    #         Terminal.text.see("end")
    #         return

    #     # endregion

    #     Terminal.button.configure(state="disabled")
    #     Terminal.text.insert("end", "\n======开始筛选======\n")
    #     Terminal.text.see("end")

    #     worker = threading.Thread(
    #         target=Execute_Main_Thread,
    #         args=(params,),
    #         daemon=True
    # )

    #     worker.start()

    # def Execute_Main_Thread(params):

    #     try:
    #         Execute_Main(params=params, log_func=log)

    #     except Exception:

    #         log("\n程序运行出错：\n")
    #         log(traceback.format_exc())

    #     finally:
    #         log("__DONE__")    

    # Check_Log_Queue()



