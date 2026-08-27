from scipy import signal
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog                                             #导入所需的库
from pathlib import Path
import bisect 
import sys

from state import Parameter_State

if __name__ == "__main__" :                                    #让这个Python 文件既可以作为主程序直接运行，也可以作为模块被其他文件导入，且导入时不会执行特定的代码块

 # region initialization

    root_window = tk.Tk()                                      #创建一个Tkinter窗口对象
    root_window.withdraw()                                     #隐藏主窗口，防止其显示在屏幕上

    # region folder choose

    # the basic source/target folder path

    importing_file_folder_path = exporting_file_folder_path = ''                #对导入和导出文件夹路径进行初始化，赋值为空字符串

    root_window.attributes("-topmost", True)                                    #使窗口保持在最前端，确保用户可以看到文件夹选择对话框
    root_window.update()                                                        #更新窗口状态，确保属性设置生效

    importing_file_folder_select = filedialog.askdirectory(parent = root_window ,
                                                            title = "请选择导入文件夹",
                                                            mustexist = True
                                                            )
    # 将对话框返回的字符串转换成 Path，便于后续使用 rglob 和路径拼接。
    importing_file_folder_path = Path(rf'{importing_file_folder_select}')
                                                                                                        #与io.py中导入文件夹选择的代码类似，弹出一个文件夹选择对话框，让用户选择导入文件夹，并将选择的路径转换为Path对象
    exporting_file_folder_select = filedialog.askdirectory(parent = root_window ,
                                                            title = "请选择导出文件夹",
                                                            mustexist = True
                                                            )
    # 保存批量特征统计表的目标目录。
    exporting_file_folder_path = Path(rf'{exporting_file_folder_select}')

    if (importing_file_folder_select == '') or (exporting_file_folder_select == ''):      #判断导入和导出文件夹路径是否为空，如果为空，则打印提示信息并退出程序

        print ( '未完全选择文件夹' )                                                                
        sys.exit()

    # endregion

    # region parameter

    params = Parameter_State()                        #创建参数数据类实例
    params.SNR_windowlength = 9                       #Savitzky-Golay平滑窗口长度；通常必须为正奇数
    params.SNR_polyorder = 3                          #平滑拟合多项式阶数；必须小于窗口长度

    Finger_Start  = 1100                                #这里是对state中的参数进行再定义，定义指纹区间的起始位置
    Finger_End  = 1800                                #指纹区间的结束位置
    Silence_Start  = 1800                             #静默区间的起始位置
    Silence_End  = 2700                               #静默区间的结束位置
    CH_Start  = 2700                                  #C-H区间的起始位置
    CH_End  = 3200                                    #C-H区间的结束位置

    # endregion

    # region parameter



# endregion

def Index_Comparison( reference_df , start_x , end_x , direction = 'inward'):       #定义一个函数Index_Comparison，用于在数据框中查找指定范围内的索引位置。

    reference_x = reference_df.loc[ : , "Raman_Shift"].tolist()                     #从DataFrame中提取"Raman_Shift"列的数据，并将其转换为列表，赋值给reference_x

    if direction == 'inward':                                                       #如果方向参数为'inward'，则使用bisect模块的bisect_left函数在reference_x中查找start_x和end_x的位置，并返回对应的索引值。       

        start_index = bisect.bisect_left( reference_x , start_x )
        end_index = ( bisect.bisect_left ( reference_x , end_x) - 1 )

    elif direction == 'outward':                                                    #如果方向参数为'outward'，则使用bisect模块的bisect_left函数在reference_x中查找start_x和end_x的位置，并返回对应的索引值，注意这里对start_index进行了减1操作。

        start_index = (  bisect.bisect_left(reference_x , start_x) - 1 )
        end_index = bisect.bisect_left ( reference_x , end_x ) 

    return (start_index , end_index )                                                #返回可用于DataFrame切片的起止索引

def SNR_Calculate( df , start  , end   , windowlength , polyorder , axis = -1):                                  #定义一个函数SNR_Calculate，用于计算信噪比（SNR）。

    smoothing_df = df.loc[start:end, 'Raman_Intensity'].to_list()                                                #从DataFrame中提取指定范围内的"Raman_Intensity"列的数据，并将其转换为列表，赋值给smoothing_df

    smoothed_df = signal.savgol_filter ( smoothing_df ,\
                                window_length=windowlength , axis=axis , polyorder = polyorder)                   #使用Savitzky-Golay滤波器对smoothing_df进行平滑处理，得到smoothed_df。
    
    noise = smoothed_df - smoothing_df                                                                            #计算平滑后的信号与原始信号之间的差值，赋值给noise。

    SNR = 10 * np.log10(np.sum(smoothed_df**2) / np.sum(noise**2))                                                #计算信噪比SNR
    return SNR                                                                                                 #返回以dB形式表示的信噪比

def Standard_Error_Calculate (second_df  , start  , end ):                                                        #定义一个函数Standard_Error_Calculate，用于计算指定范围内的标准误差。

    std_calculate_list = second_df.loc [start:end,'Raman_Intensity']                                              #从DataFrame中提取指定范围内的"Raman_Intensity"列的数据，并将其转换为列表，赋值给std_calculate_list。

    std_error = np.std(std_calculate_list)                                                                        #计算std_calculate_list的标准差，赋值给std_error。
    
    return std_error                                                                                           #返回区间强度的总体标准差（np.std默认ddof=0）

def Peak_Filter_FindPeaks( second_df , start_x , end_x , min_prominence , min_peak_len , min_distance=None ):     #定义一个函数Peak_Filter_FindPeaks，用于在指定范围内查找峰值。
    
    band_df = second_df.loc[start_x : end_x ,:].copy()                                                            #从DataFrame中提取指定范围内的所有列的数据，并创建一个副本，赋值给band_df。
    y = band_df["Raman_Intensity"].to_numpy()                                                                     #将band_df中的"Raman_Intensity"列的数据转换为NumPy数组，赋值给y。

    peaks, props = signal.find_peaks(                                                                             #使用SciPy库的find_peaks函数在y中查找峰值，返回峰值的索引和属性。
        y,
        prominence=min_prominence,                         #峰突出度下限
        width=min_peak_len,                                #峰宽下限，单位为采样点
        distance=min_distance                              #相邻峰的最小距离，None表示不限制
    )
    
    # peaks保存峰在当前波段数组中的相对位置；props保存突出度、宽度和左右边界等属性。
    return ( peaks , props )

def Filter_Data_Collect (                                                                                         #定义一个函数Filter_Data_Collect，用于收集过滤后的数据。
                         params ,
                         file_path ,
                         Finger_Start_Index , 
                         Finger_End_Index , 
                         Silence_Start_Index , 
                         Silence_End_Index , 
                         CH_Start_Index , 
                         CH_End_Index , 
                         ):

    # params：SNR计算等公共参数；file_path：当前光谱文件。
    # 三组Start/End参数均为DataFrame行索引，而不是拉曼位移值。

    imported_df = pd.read_csv (                                                                                   #从CSV文件中读取数据，创建DataFrame
                                file_path ,                                                                       
                                header = None , 
                                sep= '\t' , 
                                encoding= "utf-8" ,
                                names = ["Raman_Shift" , 'Raman_Intensity']
                                )
    
    secondary_df = imported_df [ imported_df['Raman_Shift'] > 500 ]                                                #对imported_df进行筛选，只保留"Raman_Shift"大于500的行，创建secondary_df。

    present_dict = {}                                                                                              #创建一个空字典present_dict，用于存储过滤后的数据。

    # -------------------- 指纹区特征 --------------------

    finger_df = secondary_df.loc[Finger_Start_Index :Finger_End_Index , 'Raman_Intensity']                         #从secondary_df中提取指定范围内的"Raman_Intensity"列的数据，赋值给finger_df。
    finger_fdf = secondary_df.loc[Finger_Start_Index :Finger_End_Index ,:]                                         #从secondary_df中提取指定范围内的所有列的数据，赋值给finger_fdf。
    finger_std = Standard_Error_Calculate ( secondary_df , Finger_Start_Index ,Finger_End_Index )                  #计算finger_df的标准误差，赋值给finger_std。
    finger_mean = np.mean ( finger_df )                                                                            #计算finger_df的均值，赋值给finger_mean。


    # prominence和width均设为0，先找出指纹区全部局部峰，再从中选择突出度最大的三个峰。
    finger_peaks , finger_props = Peak_Filter_FindPeaks (                                                          
                                                        secondary_df , 
                                                        Finger_Start_Index , 
                                                        Finger_End_Index , 
                                                        min_prominence= 0., 
                                                        min_peak_len= 0 ,
                                                        )
    
    # 通过“平滑信号能量/平滑残差能量”计算指纹区信噪比。
    finger_snr = SNR_Calculate ( secondary_df , Finger_Start_Index ,Finger_End_Index ,windowlength = params.SNR_windowlength , polyorder = params.SNR_polyorder)

    if len ( finger_peaks ) >= 3:                         #至少存在三个峰时才计算前三峰的完整特征

        height_list = list(finger_props['prominences'])      #所有峰的突出度列表；
        peak_position_list = []                               #初始化前三峰在属性数组中的索引列表
        
        height_array = np.asarray(finger_props['prominences'])  #转换为NumPy数组
        top3_indices = np.argsort(height_array)[-3:]             #取突出度最大的三个峰的索引
        peak_position_list = top3_indices                        #三个索引按突出度从小到大排列

        finger_peak_1_index = int (peak_position_list[0])        #前三峰中突出度最小的峰
        finger_peak_2_index = int (peak_position_list[1])        #前三峰中突出度居中的峰
        finger_peak_3_index = int (peak_position_list[2])        #整个指纹区突出度最大的峰

                                                                                           # finger_peaks记录波段内的相对数组位置；iloc将它映射到对应的Raman_Shift。
        finger_peak_1_position = finger_fdf.iloc[finger_peaks[finger_peak_1_index] , 0]
        finger_peak_2_position = finger_fdf.iloc[finger_peaks[finger_peak_2_index] , 0]
        finger_peak_3_position = finger_fdf.iloc[finger_peaks[finger_peak_3_index] , 0]

                                                                                           # prominence作为峰的绝对高度，用于后续计算C-H峰与指纹峰的峰高比。
        finger_peak_1_absolute_height = (finger_props['prominences'][finger_peak_1_index])
        finger_peak_2_absolute_height = (finger_props['prominences'][finger_peak_2_index]) 
        finger_peak_3_absolute_height = (finger_props['prominences'][finger_peak_3_index])

                                                                                            # 用峰突出度除以整个指纹区的标准差，得到无量纲的相对峰高。
        finger_peak_1_relative_height = (finger_props['prominences'][finger_peak_1_index]) / finger_std
        finger_peak_2_relative_height = (finger_props['prominences'][finger_peak_2_index]) / finger_std
        finger_peak_3_relative_height = (finger_props['prominences'][finger_peak_3_index]) / finger_std

                                                                                            # find_peaks返回的widths单位为采样点，而不是Raman_Shift单位。
        finger_peak_1_width = finger_props['widths'][finger_peak_1_index]
        finger_peak_2_width = finger_props['widths'][finger_peak_2_index]
        finger_peak_3_width = finger_props['widths'][finger_peak_3_index]

    
        
        left_nearest_pos = np.floor(finger_props["left_ips"][0]).astype(int)
        right_nearest_pos = np.ceil(finger_props["right_ips"][-1]).astype(int)               # 以检测到的第一个峰左边界和最后一个峰右边界，划分峰区及两侧非峰区。

        
        left_peak_index = finger_fdf.index [ left_nearest_pos ]                              # 将指纹区内部的相对位置换回secondary_df保留的原始行索引。
        right_peak_index = finger_fdf.index [ right_nearest_pos ]

        
        left_noise_std = Standard_Error_Calculate ( secondary_df , Finger_Start_Index , left_peak_index)# 分别计算左侧非峰区、右侧非峰区以及中间峰区的强度标准差。
        right_noise_std = Standard_Error_Calculate ( secondary_df , right_peak_index , Finger_End_Index )
        peak_std = Standard_Error_Calculate ( secondary_df , left_peak_index , right_peak_index )

    else:

        # 峰数不足三个时，无法形成完整的前三峰特征。峰形参数保留为NaN，
        # 但技术波动指标退回到整个指纹区的标准差，避免“关闭峰形限制”后
        # 仍因为NaN比较而把光谱误判为不合格。
        finger_peak_1_relative_height = finger_peak_2_relative_height = finger_peak_3_relative_height = \
        finger_peak_1_position = finger_peak_2_position = finger_peak_3_position = \
        finger_peak_1_width = finger_peak_2_width = finger_peak_3_width = np.nan
        left_noise_std = right_noise_std = peak_std = finger_std

    # 将指纹区的SNR、分区标准差以及三个主峰的特征写入当前文件结果字典。
    present_dict.update({
                'Finger_SNR':finger_snr , 
                'Finger_Peak_STD':peak_std , 
                'Finger_L_Noise_STD':left_noise_std ,  
                'Finger_R_Noise_STD':right_noise_std ,
                'Finger_1_Position': finger_peak_1_position , 
                'Finger_1_Height':finger_peak_1_relative_height , 
                'Finger_1_Width':finger_peak_1_width , 
                'Finger_2_Position':finger_peak_2_position , 
                'Finger_2_Height':finger_peak_2_relative_height , 
                'Finger_2_Width':finger_peak_2_width , 
                'Finger_3_Position':finger_peak_3_position , 
                'Finger_3_Height':finger_peak_3_relative_height ,
                'Finger_3_Width':finger_peak_3_width 
                })    

    # 静默区特征 

    silence_df = secondary_df.loc[Silence_Start_Index:Silence_End_Index , 'Raman_Intensity']  #截取静默区强度
    silence_std = Standard_Error_Calculate( secondary_df , Silence_Start_Index , Silence_End_Index )  #计算静默区整体标准差
    silence_snr = SNR_Calculate ( secondary_df , Silence_Start_Index , Silence_End_Index , windowlength = params.SNR_windowlength , polyorder = params.SNR_polyorder)  #计算静默区信噪比
    silence_mean  = np.mean ( silence_df )                  #计算静默区平均强度

    # 静默区不提取峰，只保存SNR和整体标准差。
    present_dict.update({
                'Silence_SNR':silence_snr ,
                'Silence_STD':silence_std ,
                })   

    # C-H区特征 

    CH_df = secondary_df.loc[CH_Start_Index : CH_End_Index , 'Raman_Intensity']  #截取C-H区强度
    CH_fdf = secondary_df.loc[CH_Start_Index : CH_End_Index , :]                 #保留位移、强度两列，用于位置换算
    CH_std = Standard_Error_Calculate ( secondary_df , CH_Start_Index , CH_End_Index  )  #计算C-H区整体标准差
    CH_mean = np.mean ( CH_df )                                                          #计算C-H区平均强度


    # 查找C-H区的全部局部峰，再以突出度最大的峰作为主峰。
    CH_peaks , CH_props = Peak_Filter_FindPeaks ( 
                                                secondary_df , 
                                                CH_Start_Index , 
                                                CH_End_Index , 
                                                min_prominence= 0 , 
                                                min_peak_len  = 0
                                                        )
    CH_snr = SNR_Calculate ( secondary_df , CH_Start_Index , CH_End_Index  , windowlength = params.SNR_windowlength , polyorder = params.SNR_polyorder)  #计算C-H区信噪比
    
    if len( CH_peaks ) != 0:                            #至少检测到一个峰时才提取C-H主峰特征

        main_peak_index = np.argmax(CH_props['prominences'])  #取得突出度最大峰在属性数组中的索引

        CH_peak_position = CH_fdf.iloc[CH_peaks[main_peak_index] ,0]              #C-H主峰对应的Raman_Shift
        CH_peak_absolute_height = ( CH_props['prominences'][main_peak_index])     #主峰绝对突出度
        CH_peak_relative_height = ( CH_props['prominences'][main_peak_index]) / CH_std  #主峰突出度/全区标准差
        CH_peak_width = CH_props['widths'][main_peak_index]                       #主峰宽度，单位为采样点

        # 使用全部检测峰的最左、最右边界划分C-H峰区，而不是只使用主峰边界。
        left_nearest_pos = np.floor(CH_props["left_ips"][0]).astype(int)
        right_nearest_pos = np.ceil(CH_props["right_ips"][-1]).astype(int) 

        # 以下两行可用于取得边界的Raman_Shift，目前保留为调试参考，不执行。
        # left_edge = CH_fdf.iloc[left_nearest_pos , 0]
        # right_edge = CH_fdf.iloc[right_nearest_pos , 0]

        # 把C-H区内部相对位置换回secondary_df的原始行索引。
        left_peak_index = CH_fdf.index[left_nearest_pos]
        right_peak_index = CH_fdf.index[right_nearest_pos]

        # 计算左侧非峰区、右侧非峰区和中间峰区的标准差。
        left_noise_std = Standard_Error_Calculate ( secondary_df , CH_Start_Index , left_peak_index)
        right_noise_std = Standard_Error_Calculate ( secondary_df , right_peak_index , CH_End_Index )
        peak_std = Standard_Error_Calculate ( secondary_df , left_peak_index , right_peak_index )

        # 当前判断要求存在指纹峰，下方三个绝对峰高变量只有指纹峰数>=3时才会定义。
      
        if len(finger_peaks) >= 3:

            # 分别计算C-H主峰与三个指纹主峰的绝对突出度之比。
            peak_height_ratio_1 = CH_peak_absolute_height / finger_peak_1_absolute_height
            peak_height_ratio_2 = CH_peak_absolute_height / finger_peak_2_absolute_height
            peak_height_ratio_3 = CH_peak_absolute_height / finger_peak_3_absolute_height

        else:

           
            peak_height_ratio_1 = peak_height_ratio_2 = peak_height_ratio_3 = np.nan

    else : 

        # C-H区没有检测到峰时，峰形参数和峰高比记为NaN；技术波动指标
        # 退回到整个C-H区标准差。这样在峰高/峰宽条件关闭时，仍可仅依据
        # SNR和区间波动进行技术质量筛选。
        CH_peak_position = CH_peak_relative_height = CH_peak_width = np.nan
        left_noise_std = right_noise_std = peak_std = CH_std
        peak_height_ratio_1 = peak_height_ratio_2 = peak_height_ratio_3 = np.nan

                                                                                           # 将C-H区特征和三项跨波段峰高比合并进当前文件结果字典。
    present_dict.update ({
                'CH_SNR':CH_snr , 
                'CH_Peak_STD':peak_std , 
                'CH_L_Noise_STD':left_noise_std , 
                'CH_R_Noise_STD':right_noise_std , 
                'CH_Peak_Position':CH_peak_position , 
                'CH_Peak_Height':CH_peak_relative_height ,
                'CH_Peak_Width': CH_peak_width , 
                'Peak_Height_Ratio_1': peak_height_ratio_1 ,
                'Peak_Height_Ratio_2': peak_height_ratio_2 , 
                'Peak_Height_Ratio_3': peak_height_ratio_3
                })

    return present_dict                                  #返回当前光谱文件的完整特征字典

def Print_Finding_Result():


   
    imported_file_path = list( importing_file_folder_path.rglob('*.txt') )                  # 递归获取导入目录下全部TXT；列表为空时，后面的imported_file_path[0]会引发IndexError。

    # 读取第一份光谱作为索引换算参考；默认假设所有光谱使用相同的Raman_Shift采样轴。
    reference_df = pd.read_csv( 
                                imported_file_path[0], 
                                header = None , 
                                sep= '\t' , 
                                encoding= "utf-8" ,
                                names = ["Raman_Shift" , 'Raman_Intensity']
                                )

    # region parameter transfer

                                                                                             # 将三个波段的Raman_Shift边界转换成DataFrame行索引，供所有文件复用。
    Finger_Start_Index , Finger_End_Index = Index_Comparison (reference_df , Finger_Start , Finger_End )
    Silence_Start_Index , Silence_End_Index = Index_Comparison ( reference_df , Silence_Start , Silence_End)
    CH_Start_Index , CH_End_Index = Index_Comparison ( reference_df , CH_Start , CH_End )

    rows = []                                             #保存每个光谱的特征字典

    for file_path in imported_file_path:                  #逐个提取所有TXT的特征

      
        present_dict = Filter_Data_Collect (                                                 # 对当前文件计算指纹区、静默区、C-H区以及峰高比特征。
                                            params ,
                                            file_path , 
                                            Finger_Start_Index , 
                                            Finger_End_Index ,
                                            Silence_Start_Index , 
                                            Silence_End_Index , 
                                            CH_Start_Index , 
                                            CH_End_Index ,
                                            )
        present_dict['path'] = file_path                  #记录源文件路径，便于后续关联标注结果
        rows.append(present_dict)                         #将当前结果加入总结果列表

    result_df = pd.DataFrame (rows)                       #把字典列表转换成二维表格

    # 将全部特征写入CSV；缺失值显示为nan，浮点数保留8位小数。
    result_df.to_csv ( 
                    (exporting_file_folder_path / Path('Relative_Statistical_Description.csv')) ,
                    index = True , 
                    header = True , 
                    na_rep = 'nan' ,
                    float_format = '%.8f'
                    )

if  __name__ == '__main__':

    Print_Finding_Result()                                #直接运行本文件时执行批量特征导出
