import cv2


class SVM:
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


def get_hog(image):
    descriptor = cv2.HOGDescriptor()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return descriptor.compute(gray)
