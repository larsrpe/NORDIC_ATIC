import torch
import scipy
import numpy as np
import cv2 


class ImagePDF(torch.autograd.Function):
    def __init__(self, image_file, L) -> None:
        super().__init__()
        self.image = np.asarray(cv2.imread(image_file,0))
        # self.image = Image.crop_image()
        self.h = np.shape(self.image)[0]
        self.w = np.shape(self.image)[1]
        self.dx = L / self.h
        self.dy = L / self.w

        self.pdf = self.image / (np.sum(self.image) * self.dx * self.dy)
        self.pdfdx = self.pdf_dx(self.pdf)
        self.pdfdy = self.pdf_dy(self.pdf)

    def crop_image(self):
        h, w = self.image.shape
        if h == w:
            return
        if h > w:
            self.image = self.image[(h - w) // 2 : h - (h - w) // 2, :]
        else:
            self.image = self.image[:, (w - h) // 2 : w - (w - h) // 2]

    def pdf_dx(self, pdf):
        sobel_x = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        scipy.signal.convolve2d(pdf, sobel_x)

    def pdf_dy(self, pdf):
        sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scipy.signal.convolve2d(pdf, sobel_y)

    def pos2index(self, x):
        j = int(x[1].item() / self.dx)
        i = self.h - 1 - int(x[0].item() / self.dy)
        return i, j

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.tensor(ctx.pdf[ctx.pos2index(x)])

    @staticmethod
    def backward(self, grad_output):
        x = self.saved_tensors
        i, j = self.pos2index(x)
        dx = self.pdf_dx[i, j]
        dy = self.pdf_dy[i, j]

        return grad_output * torch.tensor([dx, dy])

if __name__=="__main__":
    file_name = 'src/input_image.png'
    L = 1
    image_pdf = ImagePDF(file_name,L)
    x = torch.rand(2)*L
    print(x)
    y = image_pdf.apply(x)
    print(y)
    print('done')
