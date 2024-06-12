import cv2 as cv
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt


def calculate_sift(image_path):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE) 
    sift = cv.SIFT_create() # Inicializa el detector SIFT
    keypoints, descriptors = sift.detectAndCompute(image, None) 
    # Detección de puntos clave y cálculo de descriptores SIFT
    
    return keypoints, descriptors


def save_sift_features(image_path, keypoints, descriptors, output_directory):
    """ 
    Guarda los descriptores y los keypoints en el directorio de salida añadiendo
    la extensión .sift al nombre. Los kp se guardan con una estructura característica
    para poder recuperarse correctamente como cv2.keypoint
    """
    # Guarda las características en un archivo .sift en la carpeta de salida especificada
    filename = os.path.basename(image_path) # ruta archivo --> nombre concreto del archivo
    sift_filename = os.path.join(output_directory, filename + '.sift') # Construye la ruta completa al archivo .sift
    
    # Objeto cv2.KeyPoint --> lista de tuplas
    keypoints_info = [(kp.pt, 
                       kp.size, 
                       kp.angle, 
                       kp.response, 
                       kp.octave, 
                       kp.class_id) for kp in keypoints]

    with open(sift_filename, 'wb') as f: # Abre el archivo en modo binario para escritura
        pickle.dump((keypoints_info, descriptors), f) # Guarda los datos usando pickle


def load_sift_features(sift_path):
    """
    Carga las características SIFT desde el archivo .sift
    y las convierte a formato cv.keypoint , además de los descriptores.
    """
    keypoints = [] 
    
    with open(sift_path, 'rb') as f:
        keypoints_info, descriptors  = pickle.load(f)
    
    for coord, size, angle, response, octave, class_id in keypoints_info:
        kp = cv.KeyPoint(x=coord[0], y=coord[1], size=size, angle=angle, response=response, octave=octave, class_id=class_id)
        keypoints.append(kp)
        
    return keypoints, descriptors


def process_DB(directory, sift_directory):
    # Actualiza la base de datos
    # Actualiza la base de datos
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG") or filename.endswith(".jpeg") or filename.endswith(".PNG"):
            image_path = os.path.join(directory, filename)
            # Verifica si las características SIFT ya están calculadas y guardadas
            if os.path.exists(os.path.join(sift_directory, filename + '.sift')):
                None 
                #print(f"Características SIFT para '{filename}' ya están calculadas y guardadas.")
            else:
                print(f"\nCalculando características SIFT para '{filename}'...")
                # Calcular características SIFT para la imagen
                keypoints, descriptors = calculate_sift(image_path)
                # Guardar las características SIFT en un archivo en la carpeta de salida
                save_sift_features(image_path, keypoints, descriptors, sift_directory)
                print(f"Características SIFT para '{filename}' guardadas con éxito.")


def FLANN_matches(des1,des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50) # Inidca cuantas veces se deben
    #recorrer de forma recursiva los árboles del índice. 
    #Valor + alto = +precisión +tiempo
    
    # inicializa el descriptor
    flann = cv.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    
    # Filtrar las coincidencias utilizando el criterio de ratio de Lowe
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance: # originalmente 0.7 #0.5 bien
            good_matches.append(m)
    
    return good_matches


#==============================================================================
# MAIN C0DE    
#==============================================================================

matching = open ('cotejos.txt','w')
matching.write('LISTA DE COTEJOS')
matching.close()

n_candidatos = 5 # cambiar para indicar el n de resultados del ranking
img_query = "evidencexx.jpg"
SIFT_directory = "SIFT"
DB_directory = "BD" # Directorio de las imágenes

results = {} #diccionario de resultados

#procesamos la imagen de query
q_keypoints, q_descriptors = calculate_sift(img_query)
print(f'\nImagen de consulta {img_query} leída correctamente')

# Actualizamos la BD
process_DB(DB_directory,SIFT_directory)
print(f'\nBase de datos {SIFT_directory} actualalizada')

"""
Se calcula una puntuación (matches totales) para cada imagen de la base de datos
en comparación  con la imagen de consulta (query) y esta se guarda en un diccionario 
asociada al nombre de la imagen (filename)
"""
for filename in os.listdir(SIFT_directory):
    if filename.endswith(".sift"):
        image_path = os.path.join(SIFT_directory, filename) 
        db_keypoints, db_descriptors = load_sift_features(image_path)
        matches = FLANN_matches(db_descriptors, q_descriptors)
        similarity = len(matches) 
        results[filename[:-5]] = similarity
        
        matching = open ('cotejos.txt','a') #append, guardamos los matches porsi
        matching.write(f'\nSimilitud entre la imagen de consulta y {filename[:-5]}: {similarity}')
        matching.close()
        
print(f'\nBase de datos {SIFT_directory} leída correctamente')


"""
Se ordenan los resultados de mayor a menor por su similitud y se imprime un ranking de 
resultados + similares del nº de candidatos que indique la variable n_candidatos
"""
sorted_results = sorted(results.items(), 
                        key=lambda x: x[1], 
                        reverse=True)

print(f"\nMEJORES CANDIDATOS (ranking top {n_candidatos})") 
for rank, (filename, similarity) in enumerate(sorted_results[:n_candidatos], start=1): #n_candidatos
    print(f"Candidato {rank}: {filename} ---- Similitud: {similarity}")
    
  
for rank in range(0,n_candidatos): # imagenes para candidatos del ranking

    candidate,_ = sorted_results[rank] 
    candidate_path = os.path.join(DB_directory, candidate)
    sift_candidate_path = os.path.join(SIFT_directory, candidate + '.sift')
    
    c_keypoints, c_descriptors = load_sift_features(sift_candidate_path)
    matches = FLANN_matches(q_descriptors, c_descriptors)
    
    src_pts = np.float32([ q_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ c_keypoints[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    try:  
        print(f'\nCalculando la homografía con {candidate} ·······')
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h,w = cv.imread(img_query, cv.IMREAD_GRAYSCALE).shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
            
        img2 = cv.polylines(cv.imread(candidate_path, cv.IMREAD_GRAYSCALE),[np.int32(dst)],True,255,3, cv.LINE_AA)
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
         
        img3 = cv.drawMatches(cv.imread(img_query, cv.IMREAD_GRAYSCALE),
                              q_keypoints,img2,
                              c_keypoints,matches,None,**draw_params)
        
        plt.imshow(img3, 'gray')
        plt.show()
        
    except cv.error:
        print('~ No hay suficientes coincidencias válidas para calcular homografía')
        print('Calculando imagen de comparación ·······')
        matchesMask = [0] * len(matches)
        draw_params = dict(matchColor=(0, 255, 0), 
                            singlePointColor=(255, 0, 0),
                            matchesMask=matchesMask,
                            flags=cv.DrawMatchesFlags_DEFAULT)
        
        img3 = cv.drawMatches(cv.imread(img_query, cv.IMREAD_GRAYSCALE),
                              q_keypoints,
                              cv.imread(candidate_path, cv.IMREAD_GRAYSCALE),
                              c_keypoints, matches, None, **draw_params)
        
        plt.imshow(img3, 'gray')
        plt.show()
        


