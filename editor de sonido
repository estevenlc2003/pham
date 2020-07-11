from scipy.io import wavfile
import numpy as np
import IPython.display as ipd
from scipy.fftpack import *
import matplotlib.pyplot as plt
import math

def playAudio(file):
    """
    Muestra en pantalla el reproductor de iPython Display para un archivo de
    formato .wav.

    Parámetros
    ----------
    file: string
        Nombre del archivo en formato .wav que contiene audio en formato
        mono o estéreo.
    Retorna
    ----------
    Reproductor en pantalla de iPython con el audio estipulado

    """

    return ipd.Audio(file)

def ReadAudio(file):
    """
    Retorna la tasa de muestras por minuto y la matriz con los datos del audio}
    en formato mono o estéreo.

    Parámetros
    ----------
    file: string
        Nombre del archivo en formato .wav que contiene audio en formato
        mono o estéreo.
        
    Retorna
    --------
    List: 
        [rate,data]
    rate: int 
        Muestras por segundo
    data: numpy ndarray 
        Matriz con audio en mono o estéreo
    """
    rate,data=wavfile.read(file)
    return [rate,data]

def WriteAudio(filename,rate,matrix):
    """
    Escribe un archivo de audio .wav a partir de una matriz numpy con los datos
    del audio en mono o estéreo y la tasa de muestras por segundo.

    Parámetros
    ----------
    filename: string
        Nombre del archivo de salida .wav.
    matrix: numpy ndarray
        Matriz con audio en mono o estéreo.
    rate: int
        Tasa de muestras por minuto del audio.
    
    Retorna
    --------
    Archivo de formato wav con nombre establecido por el usuario <filename>.
    """
    wavfile.write(filename,rate,matrix)

def Speed_Rep(input_filename,speed,output_filename):
    """
    Muestra en pantalla el reproductor de audio y guarda el audio con la
    velocidad dada por el usuario para el archivo .wav estipulado.

    Parámetros
    ----------
    input_filename: string
         Nombre o localización/path del archivo .wav de entrada.
    speed: float
        Velocidad con la que se va a reproducir el audio de destino.
    output_filename: string
         Nombre o localización/path del archivo .wav de salida
    
    Retorna
    ----------
    Reproductor en pantalla de iPython con el audio con la velocidad deseada.

    """
    rate,data=ReadAudio(input_filename)
    WriteAudio(output_filename,int(rate*speed),data)
    print(f"El archivo se guardo con éxito como {output_filename}")
    return playAudio(output_filename)

def Inverse_Rep(input_filename,output_filename):
    """
    Muestra en pantalla el reproductor de audio y guarda el audio reproducido
    desde atrás en el archivo .wav estipulado.

    Parámetros
    ----------
    input_filename: string
         Nombre o localización/path del archivo .wav de entrada.
    output_filename: string
         Nombre o localización/path del archivo .wav de salida
    Retorna
    ----------
    Reproductor en pantalla de iPython con el audio hacia atrás.
    """
    
    rate,data=ReadAudio(input_filename)
    #Convertimos a mono el audio original
    data=ConvertToMono(data)
    #Leemos la matriz desde atrás usando la notación de slicing de listas
    WriteAudio(output_filename,rate,data[::-1])
    print(f"El archivo se guardo con éxito como {output_filename}")
    return playAudio(output_filename)

def ConvertToMono(data):
    """
    Retorna un array de Numpy con la matriz de audio convertida a mono con el
    mismo dtype de Numpy que el original.

    Parámetros
    ----------
    data: numpy ndarray
        Matriz de Numpy que contiene audio en formato mono o estéreo.
    
    Retorna
    ----------
    mono: numpy ndarray
        Matriz de Numpy que contiene audio en mono.
    """
    #Se procede a leer el audio
    if len(data.shape)==1:
        canales=1
    else:  
        canales=data.shape[1]

    if canales==1:
        mono=data
    #En caso de que el audio sea de formato estéreo procede a su conversión
    elif canales==2:
        mono=[]
        stereo_dtype=data.dtype
        #Se obtienen los vectores correspondientes a cada canal de audio
        l=data[:,0]
        r=data[:,1]
        #Se suma cada canal de audio para obtener uno solo
        for i in range(len(data)):
            d=(l[i]/2)+(r[i]/2)
            mono.append(d)
        mono=np.array(mono,dtype=stereo_dtype)
    return mono


def Lowpass(data,alpha):
    """
    Filtro exponencial EMA de paso bajo que recibe una matriz con audio en
    mono y retorna una matriz con audio en mono del mismo tipo con el Filtro
    aplicado.

    Parámetros
    ----------
    data: Matriz Numpy ndarray
         Matriz con los datos de un audio mono.
    alpha: float
         Factor entre 0 y 1 que determina el suavizado y aplicación del filtro.
    
    Retorna
    ----------
    filtered: numpy ndarray
        Matriz de Numpy que contiene audio en mono filtrado, con el mismo dtype
        del original.
    """
    f0=alpha*data[0]
    filtered=[f0]
    for i in range (1,len(data)):
        #y[i] := α * x[i] + (1-α) * y[i-1]
        f=alpha*data[i]+(1-alpha)*filtered[i-1]
        filtered.append(f)

    filtered=np.array(filtered,dtype=data.dtype)
    return filtered

def Highpass(data,alpha):
    """
    Filtro exponencial EMA de paso alto que recibe una matriz con audio en
    mono y retorna una matriz con audio en mono del mismo tipo con el Filtro
    aplicado.

    Parámetros
    ----------
    data: Matriz Numpy ndarray
         Matriz con los datos de un audio mono.
    alpha: float
         Factor entre 0 y 1 que determina el suavizado y aplicación del filtro.

    Retorna
    ----------
    filtered: numpy ndarray
        Matriz de Numpy que contiene audio en mono filtrado, con el mismo dtype
        del original.
    """
    f_Lowpass=Lowpass(data,alpha)
    filtered=[]
    for i in range(len(data)):
        f=data[i]-f_Lowpass[i]
        filtered.append(f)

    filtered=np.array(filtered,dtype=data.dtype)
    return filtered

def Frequency_Cutoff(type,frequency,input_filename,output_filename):
    """
    Aplica el filtro exponencial EMA de acuerdo al tipo especificado por el
    usuario (Lowpass o Highpass).

    Parámetros
    ----------
    type: string (low or high)
        Tipo de filtro (Paso bajo o paso alto).
    frequency: float
        Frecuencia (Hz) de corte para aplicación de filtro.
    input_filename: string
         Nombre o localización/path del archivo .wav de entrada.
    output_filename: string
         Nombre o localización/path del archivo .wav de salida

    Retorna
    ----------
    filtered: numpy ndarray
        Archivo de formato wav con nombre establecido por el usuario 
        <output_filename>.
    """
    rate,data=ReadAudio(input_filename)
    dt=1/rate
    alpha=(2*math.pi*dt*frequency)/((2*math.pi*dt*frequency)+1)

    print(f"α={alpha}")
    
    if type=="low":
        data_f=Lowpass(data,alpha)
    elif type=="high":
        data_f=Highpass(data,alpha)
    WriteAudio(output_filename,rate,data_f)
    print(f"El archivo se guardo con éxito como {output_filename}")

def Combinar_Audios(audio1,audio2,output_filename):
    """
    Muestra en pantalla el reproductor de audio y guarda el audio que combina 
    los dos audios de entrada.
    
    Parámetros
    ----------
    audio1: string
         Nombre o localización/path del archivo .wav de entrada.
    audio2: string
         Nombre o localización/path del archivo .wav de entrada.    
    output_filename: string
         Nombre o localización/path del archivo .wav de salida
    
    Retorna
    ----------
    Reproductor en pantalla de iPython con el audio que comnbina los audios de
    entrada.
    
    """    
    rate_1,data_1=ReadAudio(audio1)
    rate_2,data_2=ReadAudio(audio2)

    if len(data_1)>len(data_2):
        base_data=data_1.copy()
        insert_data=data_2.copy()
    else:
        base_data=data_2.copy()
        insert_data=data_1.copy()
        
    for i in range (0,int(len(insert_data))):
        base_data[i]=base_data[i]/2+insert_data[i]/2
        
    WriteAudio(output_filename,(rate_1+rate_2)//2,base_data)
    print(f"El archivo se guardo con éxito como {output_filename}")
    return playAudio(output_filename)

def FFT_Graphing(Graph_Title,data_1,rate_1,audio1_title,data_2,rate_2,audio2_title):
    """
    Grafica la transformada de fourier de dos audios, donde el eje x se
    muestra como la frecuencia en Hertz y el eje y la amplitud. Esto permite
    comparar de manera objetiva dos audios en su dominio frecuencia.

    Parámetros
    ----------
    Graph_Title: string
        Título de la gráfica.
    data_1: numpy ndarray
        Matriz con audio en mono.
    rate_1: int
        Muestras por segundo del audio.
    audio1_title: string
        Nombre a mostrar en la gráfica.
    data_2: numpy ndarray
        Matriz con audio en mono.
    rate_2: int
            Muestras por segundo del audio.
    audio2_title: string
        Nombre a mostrar en la gráfica.
    
    Retorna
    --------
    Gráfico de Matplotlib con la Transformada Rápida de Fourier de los audios de
    entrada.
    """
    plt.title(Graph_Title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")

    fft_data_1=abs(fft(data_1))
    frecs_1=fftfreq(len(fft_data_1),(1/rate_1))
    x1=frecs_1[:(len(fft_data_1)//2)]
    y1=fft_data_1[:(len(fft_data_1)//2)]

    fft_data_2=abs(fft(data_2))
    frecs_2=fftfreq(len(fft_data_2),(1/rate_2))
    x2=frecs_2[:(len(fft_data_2)//2)]
    y2=fft_data_2[:(len(fft_data_2)//2)]

    plt.plot(x1,y1,color="r",label=audio1_title)
    plt.plot(x2,y2,color="g",label=audio2_title)
    plt.legend(loc='upper right', borderaxespad=0.)
    plt.show()

def AudioGraphing(Graph_Title,data_1,rate_1,audio1_title,data_2,rate_2,audio2_title):
        """
        Grafica un audio/señal en el dominio tiempo, en el eje y se muestra la 
        señal y en el eje x el tiempo.
    
        Parámetros
        ----------
        Graph_Title: string
            Título de la gráfica.
        data_1: numpy ndarray
            Matriz con audio en mono.
        rate_1: int
            Muestras por segundo del audio.
        audio1_title: string
            Nombre a mostrar en la gráfica.
        data_2: numpy ndarray
            Matriz con audio en mono.
        rate_2: int
                Muestras por segundo del audio.
        audio2_title: string
            Nombre a mostrar en la gráfica.

        Retorna
        --------
        Gráfico de Matplotlib con los audios de entrada, en el eje x la amplitud
        y en el eje y el tiempo en segundos.
        """
        plt.title(Graph_Title)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')

        data_1=ConvertToMono(data_1)
        tiempo_1=np.arange(len(data_1))/float(rate_1)

        data_2=ConvertToMono(data_2)
        tiempo_2=np.arange(len(data_2))/float(rate_2)

        plt.fill_between(tiempo_1,data_1,color='b',label=audio1_title) 
        plt.fill_between(tiempo_2,data_2,color='m',label=audio2_title) 
        plt.legend(loc='upper right', borderaxespad=0.)
        plt.show()
