from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

app = Flask(__name__)

# Model and state holders
model = None
scaler = StandardScaler()
le_weight_class = LabelEncoder()
le_stance = LabelEncoder()
le_method = LabelEncoder()

@app.route('/train-model', methods=['POST'])
def train_model():
    global model, scaler, le_weight_class, le_stance, le_method

    data = request.get_json()
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])
    X_test = np.array(data['X_test'])

    num_classes = len(np.unique(y_train))

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_train[:len(y_pred)], y_pred, output_dict=False)

    return jsonify({
        "accuracy": float(model.evaluate(X_test, y_train[:len(y_pred)], verbose=0)[1]),
        "report": report
    })

@app.route('/predict', methods=['POST'])
def predict():
    features = np.array(request.get_json()['features'])
    probabilities = model.predict(features)[0].tolist()
    return jsonify(probabilities)

if __name__ == '__main__':
    app.run(debug=True)
