import os
"""
We need to upload files using flask and
process the uploaded image and display
it side by side
"""

from flask import Flask,render_template,request,url_for,send_from_directory,redirect
from werkzeug import secure_filename
from time import sleep
from flask import request
import PIL
from PIL import Image
from flask import jsonify
from working_on_best_model.model import model

app = Flask(__name__)
DEBUG = True

app.config['UPLOAD_FOLDER']= 'static/uploads/'

app.config['ALLOWED_EXTENSIONS'] = set(['txt','pdf','png','jpg','gif','jpeg'])


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.',1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/<imgPath>')
def index(imgPath):
    return render_template('index1.html', filepath=imgPath)

@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/process',methods=['GET','POST'])
def doImageProcessing():
    print "request = ", request.args.keys()
    print "reached preprocessing"
    imagename = 'static/uploads/'+request.args.get('name')

    # img = Image.open(imagename).convert('LA')

    # img.save('static/uploads/result.png')
    # result = {'result' : 'result.png'}
    #sleep(5)
    print imagename
    ans = model.test_image(imagename)
    print repr(ans)
    result = {'result':ans}

    return jsonify(result)

@app.route('/upload',methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    print "reachable"
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        # absolutepath = os.path.join(app.config['UPLOAD_FOLDER'], filename);
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print repr(filename)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        # return redirect(url_for('uploaded_file',
        #                         filename=filename))
        #start_processing
        return redirect(url_for('index', imgPath=filename))





# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be show after the upload
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    filepath = '../'+ app.config['UPLOAD_FOLDER']+filename
    print repr(filepath)
    return redirect(url_for('display',imgPath=filepath))
    #return send_from_directory(filepath)




if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=int("4000"),
        debug=True
    )

