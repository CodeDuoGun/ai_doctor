from flask import Flask
from flask_cors import CORS
from app.view.v1 import chat, status
# from app.app_tongue.view import view_recognize, view_tongue, view_face

# existing code omitted
app = Flask(__name__)
CORS(app, supports_credentials=True)

app.register_blueprint(chat.bp, url_prefix='/v1')
app.register_blueprint(status.bp, url_prefix='/v1')

# app.register_blueprint(view_recognize.bp, url_prefix='/v1')
# app.register_blueprint(view_tongue.bp, url_prefix='/v1')
# app.register_blueprint(view_face.bp, url_prefix='/v1')


# if __name__=="__main__":
#     app.run(host="0.0.0.0", port=5001)
