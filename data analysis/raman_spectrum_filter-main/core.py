from parameter_finding import Filter_Data_Collect                       #提取单个拉曼光谱的全部筛选特征
import pandas as pd                                                     #读取两列、制表符分隔的TXT光谱
from parameter_finding import Index_Comparison                          #把Raman_Shift范围转换为DataFrame行索引
import traceback                                                        #将异常的完整调用栈转换为日志文本
import shutil                                                           #复制文件，并尽量保留时间戳等文件元数据
from pathlib import Path                                                #路径标准化、递归搜索和目录创建

def validate_paths(input_path, output_path):

    # 在读取或删除文件之前验证目录关系，防止输出清理误伤输入数据。
    if not str(input_path).strip():
        raise ValueError("输入文件夹为空")

    if not str(output_path).strip():
        raise ValueError("输出文件夹为空")

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()   #这两句是将用户传入传出的路径标准化为绝对，并重新赋值
                                                                                                     #这整个函数在做路径的验证，确保输入输出是合法的
    if not input_path.exists():                                        #输入目录必须已经存在
        raise ValueError("输入文件夹不存在")

    if not input_path.is_dir():                                        #拒绝把普通文件当作输入目录
        raise ValueError("输入路径不是文件夹")

    if input_path == output_path:                                      #输入输出相同会导致后续删除源TXT
        raise ValueError("输出文件夹不能和输入文件夹相同")

    # 禁止互为父子目录：避免输入递归扫描到旧输出，也避免清理输出时覆盖输入树。
    if output_path in input_path.parents or input_path in output_path.parents:
        raise ValueError("输入文件夹和输出文件夹不能互为父子目录")

def Back_Filter_Excute( params , log_function , progress_function):                                 #定义了一个名为Back_Filter_Excute的函数，接受三个参数：params、log_function和progress_function。

    Input_Folder_Path = Path (params.Input_Folder_Path)                                             #导入文件路径设置
    Output_Folder_Path = Path (params.Output_Folder_Path)                                           #导出文件路径设置
    validate_paths (Input_Folder_Path , Output_Folder_Path )                                        #验证路径

    Input_Txt_Files_Collection_List = list ( Input_Folder_Path.rglob ( '*.txt' ))                   #使用 pathlib 模块，从 Input_Folder_Path 指定的文件夹开始，递归地（包括所有子文件夹） 查找所有 .txt 文件，并把结果收集成一个列表保存起来。

    if not Input_Txt_Files_Collection_List:

        log_function("\n输入文件夹中没有找到 txt 文件。\n")                                            #没有则返回
        return
    
    # 后续会针对每个文件单独完成Raman_Shift到行索引的换算。

    # -------------------- 结果容器与计数器初始化 --------------------

    Output_Txt_Files_Path_Collection = []                                                            #为输出值预留空间
    Failed_Txt_Files_Path_Collection = []                                                            #为失败的预留空间

    passed_file_count = 0                                                                            #通过检测的文件数
    processed_file_count = 0                                                                         #已处理的文件数
    total_file_count = len ( Input_Txt_Files_Collection_List )                                       #待处理的文件总数

    passed_result_finger = 0                                                                         #通过检测的指纹区文件数
    passed_result_silence = 0                                                                        #通过检测的静默区文件数
    passed_result_ch = 0                                                                             #通过检测的C-H区文件数
    passed_result_ratio = 0                                                                          #通过检测的峰高比文件数
    


    

    for txt_path in Input_Txt_Files_Collection_List:                    #依次处理递归搜索到的每个TXT

        
        # 第一列指定为Raman_Shift，第二列指定为Raman_Intensity。
        Current_Df = pd.read_csv  ( 
                                    txt_path ,  
                                    sep = '\t' , 
                                    header= None ,
                                    encoding= "utf-8" ,
                                    names = ["Raman_Shift" , 'Raman_Intensity']
                                    )
        
        # Raman_Shift区间转换为当前DataFrame的行索引；要求位移列按升序排列。

        # 三个波段分别换算，可以兼容不同文件的采样点位置不完全一致。
        Finger_Start_Index , Finger_End_Index = Index_Comparison ( Current_Df ,  params.Finger_Start , params.Finger_End)
        Silence_Start_Index , Silence_End_Index = Index_Comparison ( Current_Df ,  params.Silence_Start , params.Silence_End )
        CH_Start_Index , CH_End_Index = Index_Comparison ( Current_Df ,  params.CH_Start , params.CH_End )

       
        try:

            # 计算当前文件的SNR、峰高、峰宽、分区标准差和跨波段峰高比。
            Result_Dict = Filter_Data_Collect ( 
                                                params ,
                                                txt_path , 
                                                Finger_Start_Index , 
                                                Finger_End_Index , 
                                                Silence_Start_Index , 
                                                Silence_End_Index , 
                                                CH_Start_Index , 
                                                CH_End_Index , 
                                                )
            processed_file_count += 1                               #仅特征提取成功的文件计入已处理数

        except Exception:   
            log_function (f"\n文件处理失败，已跳过：{txt_path}\n")   #向GUI日志报告失败文件
            log_function (traceback.format_exc())                    #记录具体异常类型、位置和调用栈
            continue                                                 #异常文件不会进入通过列表或failed列表

        
        filter_result_finger = ( 
                            Result_Dict['Finger_SNR'] >= params.Finger_Min_SNR and                      #信噪比达到下限
                            Result_Dict['Finger_Peak_STD'] <= params.Finger_Peak_Max_STD and            #峰区标准差不超过上限
                            Result_Dict['Finger_L_Noise_STD'] <= params.Finger_Noise_Max_STD and        #左侧非峰区标准差不超过上限
                            Result_Dict['Finger_R_Noise_STD'] <= params.Finger_Noise_Max_STD and        #右侧非峰区标准差不超过上限
                            Result_Dict['Finger_3_Height'] >= params.Finger_Peak_Min_Height and          #最大突出峰相对高度达到下限
                            params.Finger_Peak_Max_Length >= Result_Dict['Finger_3_Width'] >= params.Finger_Peak_Min_Length  #峰宽位于上下限之间
                           )
        
       
        # if not filter_result_finger:
        #     log_function(f"\nR1未通过：{txt_path.name}\n")
        #     log_function(f"Finger_SNR = {Result_Dict['Finger_SNR']}, 阈值 >= {params.Finger_Min_SNR}\n")
        #     log_function(f"Finger_Peak_STD = {Result_Dict['Finger_Peak_STD']}, 阈值 <= {params.Finger_Peak_Max_STD}\n")
        #     log_function(f"Finger_L_Noise_STD = {Result_Dict['Finger_L_Noise_STD']}, 阈值 <= {params.Finger_Noise_Max_STD}\n")
        #     log_function(f"Finger_R_Noise_STD = {Result_Dict['Finger_R_Noise_STD']}, 阈值 <= {params.Finger_Noise_Max_STD}\n")
        #     log_function(f"Finger_3_Height = {Result_Dict['Finger_3_Height']}, 阈值 >= {params.Finger_Peak_Min_Height}\n")
        #     log_function(f"Finger_3_Width = {Result_Dict['Finger_3_Width']}, 阈值范围 {params.Finger_Peak_Min_Length} - {params.Finger_Peak_Max_Length}\n")

        # 静默区只检查信噪比下限和整体标准差上限。
        filter_result_silence = (
                            Result_Dict['Silence_SNR'] >= params.Silence_Min_SNR and             #静默区SNR达到下限
                            Result_Dict['Silence_STD'] <= params.Silence_Max_STD                  #静默区波动不超过上限
                            )
        
        # C-H区的SNR、峰区/非峰区标准差、主峰相对高度和主峰宽度必须同时合格。
        filter_result_ch = (
                            Result_Dict['CH_SNR'] >= params.CH_Min_SNR and                          #C-H区SNR达到下限
                            Result_Dict['CH_Peak_STD'] <= params.CH_Peak_Max_STD and                #C-H峰区标准差不超过上限
                            Result_Dict['CH_L_Noise_STD'] <= params.CH_Noise_Max_STD and            #左侧非峰区标准差不超过上限
                            Result_Dict['CH_R_Noise_STD'] <= params.CH_Noise_Max_STD and            #右侧非峰区标准差不超过上限
                            Result_Dict['CH_Peak_Height'] >= params.CH_Peak_Min_Height and           #C-H主峰相对高度达到下限
                            Result_Dict['CH_Peak_Width'] >= params.CH_Peak_Min_Length               #C-H主峰宽度达到下限（无最大值限制）
                           )
        
        # C-H主峰与指纹区最大峰的绝对突出度比值达到下限。
        filter_result_ratio = Result_Dict['Peak_Height_Ratio_3'] >= params.Peak_Height_Ratio

        # 分别统计四组独立条件的通过数量，用于任务结束时计算单项通过率。
        if filter_result_finger:
            passed_result_finger += 1                                #指纹区通过计数
        if filter_result_silence:
            passed_result_silence += 1                               #静默区通过计数
        if filter_result_ch:
            passed_result_ch += 1                                    #C-H区通过计数
        if filter_result_ratio:
            passed_result_ratio += 1                                 #峰高比通过计数

        # 每成功处理10个文件更新一次进度；最后一个成功文件也会触发更新。
       
        if processed_file_count % 10 == 0 or processed_file_count == total_file_count:

            # progress_function由GUI传入；后台线程只写队列，不直接操作Tk控件。
            progress_function(
                            '进度：{}/{}，{:.2%}      '.format(
                            processed_file_count , 
                            total_file_count, 
                            processed_file_count/total_file_count )
                            )
            
        # 最终通过要求4四组条件全部为True。
        if filter_result_finger and filter_result_silence and filter_result_ch and filter_result_ratio:

            passed_file_count += 1                                  #最终通过文件数量加一
            Output_Txt_Files_Path_Collection.append (txt_path)       #记录源路径，循环结束后统一复制

        else:

            Failed_Txt_Files_Path_Collection.append (txt_path)       #任一条件失败就加入未通过列表

    
    # 在复制本轮结果前，递归删除输出目录下全部旧TXT，包括旧failed目录中的TXT。
    # unlink是直接删除而非移动到回收站；validate_paths用于确保这里不会清理输入目录。
    delete_file_type = '*.txt'                                      #限定只清理TXT，其他类型文件保留
    for file_need_delete in Output_Folder_Path.rglob( delete_file_type ):
        if file_need_delete.is_file():                              #防止对非普通文件调用unlink
            file_need_delete.unlink()                               #删除上一轮产生的TXT结果

    # -------------------- 复制通过文件 --------------------
    for passed_txt_path in Output_Txt_Files_Path_Collection:        #逐个复制最终通过筛选的光谱

        relative_path = passed_txt_path.relative_to(Input_Folder_Path)  #取得文件相对于输入根目录的层级
        export_path = Output_Folder_Path / relative_path                #在输出根目录下复现原目录结构
        export_path.parent.mkdir(parents=True, exist_ok=True)            #递归创建目标父目录；已存在时不报错

        shutil.copy2 ( passed_txt_path , export_path )                   #复制内容并尽量保留时间戳等元数据

    # 未通过文件统一放入输出根目录的failed子目录。
    Failed_Folder_Path = Output_Folder_Path / 'failed'

    # -------------------- 复制未通过文件 --------------------
    for failed_txt_path in Failed_Txt_Files_Path_Collection:

        relative_path = failed_txt_path.relative_to(Input_Folder_Path)  #保留相对于输入目录的原始层级
        export_path = Failed_Folder_Path / relative_path                #把层级放到failed目录之下
        export_path.parent.mkdir(parents=True, exist_ok=True)            #确保目标目录存在

        shutil.copy2 ( failed_txt_path , export_path )                   #复制未通过光谱并保留元数据

    # -------------------- 输出总体统计日志 --------------------
    # 报告输入TXT总数、最终通过数和总体通过率；异常跳过文件仍包含在总数分母中。
    log_function(
            
        '\n原有{}个样本；现有{}个文件达到了标准；通过率{:.2%}'.format(
                                                                        
        len(Input_Txt_Files_Collection_List) ,                         #输入目录中找到的全部TXT数量
        passed_file_count ,                                            #同时通过四组条件的文件数
        passed_file_count / len (Input_Txt_Files_Collection_List)      #最终总通过率
        
    ))

   
    log_function (

        '\n通过率：\n\
        指纹区R1通过率{:.2%}；\n\
        静默区R2通过率{:.2%}；\n\
        C-H峰R3通过率{:.2%}；\n\
        峰比R4通过率{:.2%}'.format(
                                passed_result_finger/total_file_count ,  #指纹区单项通过率
                                passed_result_silence/total_file_count , #静默区单项通过率
                                passed_result_ch/total_file_count ,      #C-H区单项通过率
                                passed_result_ratio/total_file_count     #峰高比单项通过率
                            )

    )
