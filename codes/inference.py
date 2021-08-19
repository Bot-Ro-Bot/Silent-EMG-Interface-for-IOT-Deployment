from config import *

class _Inference:
    
    model = None
    _instance = None

    def features(self,path):
        data = parser([path],DEPLOY=True,LENGTH=LENGTH)["data"]
        data = signal_pipeline(data)
        return data

    def predict(self,path):
        feat = self.features(path)[0]
        feat = feat[np.newaxis,...]
        prediction = np.argmax(self.model.predict(feat))
        return SENTENCES[int(prediction)]

def Inference():
    if _Inference._instance is None:
        # if instance is not created, create one
        _Inference._instance = _Inference()
        try:
            _Inference.model = keras.models.load_model(os.path.join(MODEL_DIR,"model1D.h5"))
        except Exception as ex:
            print("Model Load Failed due to: ",ex)
    return _Inference._instance


if __name__ == "__main__":
    inf = Inference()
    print(inf.predict("../test/test1.txt"))