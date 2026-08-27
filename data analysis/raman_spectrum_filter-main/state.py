from dataclasses import dataclass     #从dataclasses包中下载dataclass

@dataclass                            #装饰器
class Parameter_State:                              #定义一个类

    Input_Folder_Path : str = ''
    Output_Folder_Path : str = ''

    SNR_windowlength : int = 9 
    SNR_polyorder : int = 3

    # 细胞光谱清洗参数：不同衰老状态的正常细胞峰形差异较大，
    # 不再用峰宽、峰高或跨区峰比限定生物学形态，只用SNR与各区波动
    # 排除宇宙射线尖峰、强噪声和明显采集失败。
    Finger_Peak_Min_Length : float = 0
    Finger_Peak_Min_Height : float = 0
    Finger_Peak_Max_STD : float = 500
    Finger_Min_SNR : float = 35
    Finger_Noise_Max_STD : float = 500
    Finger_Peak_Max_Length : float = 10000
                                                                           #对应的参数进行初始化，设置默认值，为后续的参数传入做准备
    Silence_Min_SNR : float = 35
    Silence_Max_STD : float = 50

    CH_Peak_Min_Length : float= 0
    CH_Peak_Min_Height : float = 0
    # C-H峰区的标准差会随正常宽峰的绝对强度增大；P7全量复检后以700
    # 作为保护上限，避免500误删强而完整的C-H峰。
    CH_Peak_Max_STD : float = 700
    CH_Min_SNR : float = 35
    CH_Noise_Max_STD : float = 500

    Finger_Start : float = 600
    Finger_End : float = 1800
    Silence_Start : float = 1800
    Silence_End :float = 2700
    CH_Start : float = 2700
    CH_End : float = 3200

    Peak_Height_Ratio : float = 0





# region default set 

# finger_2.Start_Var.set( 1100 )
# finger_2.End_Var.set( 1800 )
# silence_2.Start_Var.set(1800)
# silence_2.End_Var.set(2700)
# C_H_2.Start_Var.set(2700)
# C_H_2.End_Var.set(3200)

# finger_3.Min_Peak_Height.set(0.5)
# finger_3.Min_Peak_Length.set(10)
# finger_3.Min_SNR_Var.set(36)
# finger_3.Peak_Max_Standard_Error.set(500)
# finger_3.Noise_Max_Standard_Error.set(500)

# silence_3.Min_SNR_Var.set(37)
# silence_3.Max_Standard_Error.set(253)

# C_H_3.Min_Peak_Height.set(2)
# C_H_3.Min_Peak_Length.set(20)
# C_H_3.Min_SNR_Var.set(0)
# C_H_3.Peak_Max_Standard_Error.set(5000)
# C_H_3.Noise_Max_Standard_Error.set(5000)

# finger_3.Peak_Height_Ratio.set(2)
# C_H_3.Peak_Height_Ratio.set(2)
