import cv2

# acha quadrado
# binariza imagem em busca dos dados
# marca onde tem quadrado
# analisa dado por dado
# conta as bolas pretas

if __name__ == "__main__":
    img = cv2.imread("dados.jpg", cv2.IMREAD_COLOR)
    cv2.imshow("Original", img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Cinza", gray_image)
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binarizado", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = gray_image.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    for contour in contours:
        cv2.imshow("Imagem", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
