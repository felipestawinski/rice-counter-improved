#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  '60.bmp'

# TODO: ajuste estes parâmetros!
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 15
LARGURA_MIN = 15
N_PIXELS_MIN = 30

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    # TODO: escreva o código desta função.
    # Dica/desafio: usando a função np.where, dá para fazer a binarização muito
    # rapidamente, e com apenas uma linha de código!
    return np.where(img >= threshold, 1.0, 0.0)

    # método "alternativo" (mais lento):
    # rows, cols, channels = img.shape
    # for row in range(rows):
    #     for col in range(cols):
    #         if img[row, col] >= threshold:
    #             img[row,col] = 1.0
    #         else:
    #             img[row,col] = 0.0
    # return img

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):

    # TODO: escreva esta função.
    # Use a abordagem com flood fill recursivo.
    rows, cols = img.shape
    label = 2
    outputList = []

    for row in range(rows):
        for col in range(cols):
            if img[row,col] == 1.0:
                n_pixels = 0
                info = flood_fill (img, row, col, label, n_pixels) 
                component = {
                    'label': label,
                    'n_pixels': info['n_pixels'],
                    'T': info ['T'],
                    'L': info ['L'],
                    'B': info ['B'],
                    'R': info ['R']
                }

                if (component['n_pixels'] >= n_pixels_min):
                    if component['B'] - component['T'] >= altura_min and \
                       component['R'] - component['L'] >= largura_min:
                        outputList.append(component)
                        label += 1
    return outputList


def flood_fill(img, y0, x0, label, n_pixels):
    img[y0, x0] = label
    n_pixels += 1
    rows, cols = img.shape
    
    # initialize bounds
    top = y0
    left = x0
    bottom = y0
    right = x0
    
    # check neighbors
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dy, dx in directions:
        y, x = y0 + dy, x0 + dx
        
        if (0 <= y < rows and 0 <= x < cols and img[y, x] == 1.0):
            sub_result = flood_fill(img, y, x, label, 0)
            
            top = min(top, sub_result['T'])
            left = min(left, sub_result['L'])
            bottom = max(bottom, sub_result['B'])
            right = max(right, sub_result['R'])
            
            #update pixel count
            n_pixels += sub_result['n_pixels']
    
    # return component information
    return {
        'T': top,
        'L': left,
        'B': bottom,
        'R': right,
        'n_pixels': n_pixels
    }


#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
        
    img_blur = cv2.boxFilter(img, 5)
    cv2.imshow (f'{INPUT_IMAGE} - blur', img)
    img = binariza(img_blur, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite(f'{INPUT_IMAGE} - binarizada.png', img*255)

    start_time = timeit.default_timer()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow('02 - out', img_out)
    cv2.imwrite('02 - out.png', img_out*255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main() 

#===============================================================================
