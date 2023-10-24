import os
import tempfile
from flask import Flask, request, redirect, render_template, url_for, send_file
from skimage import io
import base64
import glob
from skimage.transform import resize
import numpy as np
from keras.models import load_model


app = Flask(__name__, template_folder="../templates/")
main_html = """
<html>
<head></head>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

   function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
   }

  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");


      numero = getRndInteger(0, 10);
      letra = ['ア', 'イ', 'ウ', 'エ', 'オ'];
      random = Math.floor(Math.random() * letra.length);
      aleatorio = letra[0];

      document.getElementById('mensaje').innerHTML  = 'Dibujando un ' + aleatorio;
      document.getElementById('numero').value = aleatorio;

      $('#myCanvas').mousedown(function (e) {
          mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
      });

      $('#myCanvas').mousemove(function (e) {
          if (mousePressed) {
              Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
          }
      });

      $('#myCanvas').mouseup(function (e) {
          mousePressed = false;
      });
        $('#myCanvas').mouseleave(function (e) {
          mousePressed = false;
      });
  }

  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 11;
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.closePath();
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }

  function clearArea() {
      // Use the identity matrix while clearing the canvas
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
     var canvas = document.getElementById('myCanvas');
     document.getElementById('myImage').value = canvas.toDataURL();
  }



</script>
<body onload="InitThis();">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript" ></script>
    <div align="center">
        <h1 id="mensaje">Dibujando...</h1>
        <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
        <br/>
        <br/>
        <button onclick="javascript:clearArea();return false;">Borrar</button>
    </div>
    <div align="center">
      <form method="post" action="upload" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
      <input id="numero" name="numero" type="hidden" value="">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Enviar">
      </form>
    </div>
</body>
</html>

"""

# make routes
@app.route("/make/")
def make_main():
    return(main_html)

@app.route('/make/upload', methods=['POST'])
def make_upload():
    try:
        # check if the post request has the file part
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        aleatorio = request.form.get('numero')
        print(aleatorio)
        with tempfile.NamedTemporaryFile(delete = False, mode = "w+b", suffix='.png', dir=str(aleatorio)) as fh:
            fh.write(base64.b64decode(img_data))
        #file = request.files['myImage']
        print("Image uploaded")
    except Exception as err:
        print("Error occurred")
        print(err)

    return redirect("/make/", code=302)

@app.route('/make/prepare', methods=['GET'])
def make_prepare_dataset():
    images = []
    d = ['ア', 'イ', 'ウ', 'エ', 'オ']
    digits = []
    for digit in d:
        filelist = glob.glob('{}/*.png'.format(digit))
        images_read = io.concatenate_images(io.imread_collection(filelist))
        images_read = images_read[:, :, :, 3]
        digits_read = np.array([digit] * images_read.shape[0])
        images.append(images_read)
        digits.append(digits_read)
    images = np.vstack(images)
    digits = np.concatenate(digits)
    np.save('X.npy', images)
    np.save('y.npy', digits)
    return "OK!"

@app.route('/make/X.npy', methods=['GET'])
def make_download_X():
    return send_file('X.npy')

@app.route('/make/y.npy', methods=['GET'])
def make_download_y():
    return send_file('y.npy')


# main routes

@app.route("/")
def main():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('modelo_entrenado.h5')
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        print()
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix='.png', dir=str('prediccion')) as fh:
            print("start reading file")
            fh.write(base64.b64decode(img_data))
            tmp_file_path = fh.name
            print("done reading")
        imagen = io.imread(tmp_file_path)
        imagen = imagen[:, :, 3]
        size = (28, 28)
        image = imagen / 255.0
        im = resize(image, size)
        im = im[:, :, np.newaxis]
        im = im.reshape(1, *im.shape)
        salida = model.predict(im)[0]
        print("paso segunda parte")
        os.remove(tmp_file_path)
        nums = salida*100
        numeros_formateados = [f'{numero:.2f}' for numero in nums]
        cadena_formateada = ', '.join(numeros_formateados)
        return redirect(url_for('show_predictions', nums=cadena_formateada, img_data=img_data))
    except:
        print("Error occurred")

    return redirect("/", code=404)

@app.route('/predicciones')
def show_predictions():
    nums = request.args.get('nums')
    img_data = request.args.get('img_data')
    componentes = nums.split(', ')
    nums = [float(componente) for componente in componentes]
    frutas = ['ア', 'イ', 'ウ', 'エ', 'オ']
    if img_data is not None:
        return render_template('Prediccion.html', nums=nums, frutas=frutas, img_data=img_data)
    else:
        return redirect("/", code=302)

if __name__ == "__main__":
    digits = ['ア', 'イ', 'ウ', 'エ', 'オ']
    for d in digits:
        if not os.path.exists(str(d)):
            os.mkdir(str(d))
    app.run()

