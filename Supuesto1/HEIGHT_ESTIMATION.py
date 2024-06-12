"""
SINGLE-VIEW METROLOGY 
Created on Sat May 4 18:00:06 2024
@author: aidan

FUNCION DE ESTIMACION DE ALTURA BASADA EN: 
Function Name: computeCameraHeight 
Gets the camera height from the vanishing points and the height of a
reference object with known botton and top image projections
REFERENCE: Criminisi, Reid and Zisserman. Single View Metrology. 1999
 Revision: v1.0$ 
 Author: Carlos Beltran-Gonzalez$
 Email: carlos.beltran@iit.it
Istituto Italiano di Tecnologia
Pattern Analysis and Computer Vision
Date:  May 26, 2020$
Copyright (C) 2020 Carlos Beltran-Gonzalez
CopyPolicy: GNU Lesser General Public License v3
"""

import cv2
import ctypes
import numpy as np

def compute_element_height(v1, v2, v3, ref_points, measure_points, Zr)-> float:
    """
    Parameters
    ----------
    v1, v2, v3: 
        TYPE np.array([x,y,w]) --> np.array([x,y,1])
        DESCRIPTION. v1 y v2 son los vanishing points donde convegen
        las lineas paralelas horizontales (ejes x e y).v3 es el vertical 
        vanishing point donde convergen las lineas verticales ("eje z").

    ref_points : 
        TYPE [np.array([x,y,1]),np.array([x,y,1])]
        DESCRIPTION.Puntos extremos del elemento de referencia,bottom y top,
        respectivamente.
        
    measure_points : 
        TYPE [np.array([x,y,1]),np.array([x,y,1])]
        DESCRIPTION.Puntos extremos del objeto que queremos medir,bottom y 
        top,respectivamente.
        
    Zr : 
        TYPE float
        DESCRIPTION.Altura conocida del elemento de referencia, en cm.

    Returns
    -------
    Zx : 
        TYPE float
        DESCRIPTION. Altura estimada del objeto a medir, en cm.
    
    Zcam : 
        TYPE float
        DESCRIPTION. Altura de la camara.
    """
    global hoznor
    
    b_x, t_x = measure_points # puntos forma homography [x,y,w], w=1
    b_r, t_r = ref_points #puntos del obj de referencia

    hoz = np.cross(v1, v2) # vanishing line o horizonte
    
    hoznor = hoz / np.linalg.norm(hoz) # Normalize the horizon line
    print('Cálculo vanishing line o horizon: ')
    print('v1: ', v1)
    print('v2: ', v2)
    print('hoz = np.cross(v1, v2)')
    print('Horizon: ', hoznor)
    
    
    # Compute the unknown scaling factor using known target height
    alpha = (-np.linalg.norm(np.cross(b_r, t_r))) / (np.dot(hoznor, b_r) * np.linalg.norm(np.cross(v3, t_r)) * Zr)

    print('\nCálculo factor de escala (alpha): ')
    print('Zr: ', Zr)
    print('br: ', b_r)
    print('tr: ', t_r)
    print('v3: ', v3)
    print('horizon: ', hoznor)
    print("alpha = (-||bx x tx||) / (||horizon · bx|| · ||v3 x tx|| · Zr)")
    print('Alpha: ', alpha)
    
    
    # Compute the suspect height 
    Zx = (-np.linalg.norm(np.cross(b_x, t_x))) / (np.dot(hoznor, b_x) * np.linalg.norm(np.cross(v3, t_x)) * alpha)
    
    print('\nCálculo altura:')
    print('bx: ', b_x)
    print('tx: ', t_x) 
    print('v3: ', v3)
    print('alpha: ', alpha)
    print('horizon: ', hoznor)
    print("Zx = (-||bx x tx||) / (||horizon · bx|| · ||v3 x tx|| · alpha)")
    print('Height: ', Zx)
    
    # Compute camera height
    #Zcam = -(np.linalg.inv(np.dot(hoznor, v3))) / alpha

    return Zx, hoznor


def draw_horizon_line(img, hoznor):
    # Dimensiones de la imagen
    height, width, _ = img.shape
    
    # Puntos de intersección con los bordes verticales de la imagen
    x1 = 0
    y1 = int((-hoznor[2] - hoznor[0]*x1) / hoznor[1])
    x2 = width - 1
    y2 = int((-hoznor[2] - hoznor[0]*x2) / hoznor[1])
    
    # Dibujar la línea de horizonte
    cv2.line(img, (x1, y1), (x2, y2), (5, 215, 255), 2)
    
    return img

def load_and_scale_image(image_path):
    """
    Adapta la imagen al tamaño de la pantalla
    """
    global proporcion
    user32 = ctypes.windll.user32 #Obtiene dimensiones pantalla
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Escalar la imagen para ajustarla a las dimensiones de la pantalla
    scaling_factor = min(screen_width / image.shape[1], screen_height / image.shape[0])
    scaled_width = int(image.shape[1] * scaling_factor * proporcion)  # Cambiar el 1 para
    scaled_height = int(image.shape[0] * scaling_factor * proporcion)  # reducir el tamaño
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    
    return scaled_image

            
def paralel_lines(event, x, y, flags, param):
    """
    all_lines <-- list[[list],...]
    Toma los parametros proporcionados por el teclado y las variables
    globales lines que solo se utiliza en este fragmento de codigo. 
    Guarda las lineas paralelas como una lista formada por dos puntos
    en la lista all_lines.
    """
    global lines, color_index, all_lines
    
    if event == cv2.EVENT_LBUTTONDOWN:
        lines.append((x, y))
        
        if len(lines) == 2: #par de lineas
            cv2.line(img, lines[0], lines[1], colors[color_index], 2)
            print(f"Nueva línea: {lines}")
            color_index += 1
            all_lines.append(lines)
            lines = []  # Limpiar la lista de puntos después de dibujar una línea
            if color_index == len(colors):
                cv2.imshow('Image', img)  # Mostrar la imagen antes de cerrar la ventana
                cv2.waitKey(2000)  # Esperar 3 segundos (2000 milisegundos)
                cv2.imwrite('imagen_con_lineas.jpg', img)  # Guardar la imagen con líneas dibujadas
                cv2.destroyAllWindows()
                return
            cv2.imshow('Image', img)  # Actualizar la imagen después de dibujar las líneas
            
def calc_vanishing_point(line1, line2)-> (int,int):  
    """
    Parameters
    ----------
    line1 : list[(int,int),(int,int)]
        DESCRIPTION.Linea compuesta por dos puntos xy.
        
    line2 : list[(int,int),(int,int)]
        DESCRIPTION.Linea compuesta por dos puntos xy.

    Returns
    -------
    v_point : (int,int)
        DESCRIPTION.Punto de interseccion de las dos lineas no paralelas en 
        la imagen. En este caso constituye el punto de fuga, vanishing points 
        de las lineas paralelas. 
    """
    
    # Descomponemos las lineas en sus puntos
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    # Calcula las coordenadas de intersección (fórmula de intersección de dos rectas)
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    v_point = (int(round(px)), int(round(py))) # puntos de la imagen = nº enteros
    
    return v_point
    
def get_measure_points(event, x, y, flags, param):
    global measure_points
    is_line_selected = False
    
    if not is_line_selected and event == cv2.EVENT_LBUTTONDOWN:
        measure_points.append((x, y))
        if len(measure_points) == 2:
            cv2.line(img, measure_points[0], measure_points[1], (130, 0, 75), 2)
            cv2.circle(img, measure_points[0], 5, (130, 0, 75), 1, )  # Dibujar el primer punto en rojo
            cv2.circle(img, measure_points[1], 5, (130, 0, 75), 1)  # Dibujar el segundo punto en rojo
            cv2.imshow('Image', img) # Mostrar la imagen con la línea dibujada
            print("Puntos altura a medir:")
            print("Bottom ------", measure_points[0])
            print("Top ---------", measure_points[1])
            cv2.imwrite('seleccion_altura.jpg', img)
            is_line_selected = True  # Marcar que se ha seleccionado una línea
            cv2.waitKey(2000)  # Esperar 2 segundos
            cv2.destroyAllWindows()  # Cerrar la ventana después de 2 segundos

def get_reference_points(event, x, y, flags, param):
    global ref_points
    is_line_selected = False
    
    if not is_line_selected and event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))
        if len(ref_points) == 2:
            cv2.line(img, ref_points[0], ref_points[1], (238, 104, 123), 2)
            cv2.circle(img, ref_points[0], 5, (238, 104, 123), 1)  # Dibujar el primer punto en rojo
            cv2.circle(img, ref_points[1], 5, (238, 104, 123), 1) 
            cv2.imshow('Image', img) # Mostrar la imagen con la línea dibujada
            print("Puntos altura de referencia:")
            print("Bottom ------", ref_points[0])
            print("Top ---------", ref_points[1])
            is_line_selected = True  # Marcar que se ha seleccionado una línea
            cv2.imwrite('referencias.jpg', img)
            cv2.waitKey(2000)  # Esperar 2 segundos
            cv2.destroyAllWindows()  # Cerrar la ventana después de 2 segundos


def homogeneous_vector(point) -> np.array:
    """
    Punto (x,y) --> vector homogéneo [x, y, 1].
    Parametro: tuple (point) // Returns: list
    """
    x, y = point
    return np.array([x, y, 1])


def get_float_input(prompt): 
    """
    Parameters
    ----------
    prompt : str
        DESCRIPTION.Mensaje a imprimir en la consola para pedir al 
        usuario que introduzca un numero float

    Returns
    -------
    value : float
        DESCRIPTION.Valor introducido en la consola.
        
    Este subprograma pide al usuario, a través de la consola, que 
    introduzca la altura del objeto de referencia
    
    """
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Por favor, introduce un número válido.")


def draw_horizon_line_extended(img, hoznor):
    # Dimensiones de la imagen original
    height, width, _ = img.shape
    
    # Crear un lienzo blanco más grande
    border_size = 500  # Tamaño del borde blanco
    canvas_height = height + 2 * border_size
    canvas_width = width + 2 * border_size
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255  # Lienzo blanco
    
    # Colocar la imagen original en el centro del lienzo
    canvas[border_size:border_size+height, border_size:border_size+width] = img
    
    # Puntos de intersección con los bordes verticales del lienzo extendido
    x1 = 0
    y1 = int((-hoznor[2] - hoznor[0] * (x1 - border_size)) / hoznor[1]) + border_size
    x2 = canvas_width - 1
    y2 = int((-hoznor[2] - hoznor[0] * (x2 - border_size)) / hoznor[1]) + border_size
    
    # Dibujar la línea de horizonte
    cv2.line(canvas, (x1, y1), (x2, y2), (5, 215, 255), 2)
    
    return canvas


def draw_text(img, text, top_point, 
              font=cv2.FONT_HERSHEY_SIMPLEX, 
              font_scale=0.5, 
              background_color=(255, 255, 255), 
              text_color=(0, 0, 0), 
              thickness=1):
    
    # Obtener el tamaño del texto
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Definir el tamaño del cuadrado blanco que contendrá el texto
    box_width = text_size[0] + 10
    box_height = text_size[1] + 10

    # Calcular las coordenadas del cuadrado blanco
    x1 = top_point[0]
    y1 = top_point[1] - box_height -20
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Dibujar el cuadrado blanco
    cv2.rectangle(img, (x1, y1), (x2, y2), background_color, -1)

    # Posición para el texto (centrado dentro del cuadrado blanco)
    text_x = x1 + 5  # Ajuste para centrar horizontalmente
    text_y = y1 + text_size[1] + 5  # Ajuste para centrar verticalmente

    # Dibujar el texto dentro del cuadrado blanco
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

    return img

    
#=============================================================================
# MAIN CODE
#=============================================================================

# VARIABLES GLOBALES
# ==================

lines = [] # acumula los pares de lineas --> solo para el subprograma
all_lines = [] # lineas paralelas = 6 elementos formados por 2 puntos
measure_points = [] # [bx,tx]
ref_points = [] # [br,tr]
colors = [(0, 0, 200), (0, 0, 200), # Dos líneas de cada color eje x
          (50, 255, 50), (50, 205,50), #                          eje y 
          (255, 0, 0), (255, 0, 0)] #                          eje z
color_index = 0
proporcion = 0.9
n_image = 0

print()
print('Seleccionar en la imagen las lineas paralelas en el orden X Y Z\n')
print("PARALEL LINES")
print("-------------")

img = load_and_scale_image('park1.bmp') # Cargar y escalar la imagen
cv2.imshow('Image', img) # Mostrar la imagen

# Configurar el detector de clics en la ventana de la imagen
cv2.setMouseCallback('Image', paralel_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Guardardar cada linea de all_lines en una variable
x_line1 = all_lines[0]
x_line2 = all_lines[1]
y_line1 = all_lines[2]
y_line2 = all_lines[3]
z_line1 = all_lines[4]
z_line2 = all_lines[5]
                          
# Calculo de vanishing points --> vector homogeneo                                                                
v1 = homogeneous_vector(calc_vanishing_point(x_line1, x_line2)) 
v2 = homogeneous_vector(calc_vanishing_point(y_line1, y_line2)) 
v3 = homogeneous_vector(calc_vanishing_point(z_line1, z_line2))

print('\nVANISHING POINTS')
print('----------------')
print('Eje x: horizontal vanishing point ------', (v1[0],v1[1]))
print('Eje y: horizontal vanishing point ------', (v2[0],v2[1]))
print('Eje z: vertical vanishing point   ------', (v3[0],v3[1]))

print('\nVANISHING POINTS HOMOGENEUS VECTORS')
print('-----------------------------------')
print('v1 ------', v1)
print('v2 ------', v2)
print('v3 ------', v3)

# PUNTOS DEL ELEMENTO DE REFERENCIA EN LA IMAGEN
print('\nSELECCION ALTURA DE REFERENCIA')
print('------------------------------')
print('Seleccione en la imagen el objeto de referencia: ')
print('1º Seleccione punto inferior (bottom)')
print('2º Seleccione punto superior (top)','\n')

cv2.imshow('Image', img) # Mostrar la imagen
cv2.setMouseCallback('Image', get_reference_points) # Configura detector clics
cv2.waitKey(0) # Esperar a que se cierre la ventana
cv2.destroyAllWindows()

ref_points = [homogeneous_vector(ref_points[0]),homogeneous_vector(ref_points[1])]

print('\nSELECCION ALTURA DE REFERENCIA (Zr)')
print('-----------------------------------')
ref_height = get_float_input("Introduce la altura del objeto de referencia en cm: ")

# Pintar altura
img = draw_text(img, text='{:.2f} cm'.format(ref_height), top_point =ref_points[1])
cv2.imshow('Image', img)
cv2.imwrite('referencias.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Seleccionar imagenes a analizar 
total_height = []
frames = ['21.bmp', '22.bmp', '23.bmp', '24.bmp', '25.bmp', '26.bmp']

for frame in frames:
    
    n_image += 1
    measure_points = [] # [bx,tx]
    
    img = load_and_scale_image(frame) # Cargar y escalar la imagen
    
    # PUNTOS DE LA ALTURA A MEDIR EN LA IMAGEN
    print('\nSELECCION ALTURA A MEDIR')
    print('------------------------')
    print('Seleccione en la imagen la altura que desea medir: ')
    print('1º Seleccione punto inferior (bottom)')
    print('2º Seleccione punto superior (top)','\n')
    
    
    cv2.imshow('Image', img) # Mostrar la imagen
    cv2.setMouseCallback('Image', get_measure_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    measure_points = [homogeneous_vector(measure_points[0]),homogeneous_vector(measure_points[1])]
    
    print(f'\nCÁLCULO DE LA ALTURA (Zx) EN {frame}')
    print('-------------------------')
    height, horizon = compute_element_height(v1, v2, v3, ref_points, measure_points, ref_height)
    total_height.append(height)
    print('\nEL SUJETO MEDIDO TIENE UNA ESTATURA DE: {:.2f}'.format(height), 'cm ----', round(height / 100, 2), 'm' )
    
    img = draw_text(img, text='{:.2f} cm'.format(height), top_point =measure_points[1])
    cv2.imshow('Image', img)
    cv2.imwrite(f'seleccion_altura{n_image}.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Dibujar la línea de horizonte en el lienzo extendido
    img_horizon_line = draw_horizon_line_extended(img, horizon)
    # Mostrar y guardar la imagen con la línea de horizonte
    cv2.imwrite(f'result{n_image}.jpg', img_horizon_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(total_height)
mean_height = sum(total_height)/len(total_height)
print('\n', mean_height)
    
