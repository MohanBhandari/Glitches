from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

dic = {0: '1080Lines', 
	 1: '1400Ripples', 
	 2: 'Air_Compressor', 
	 3: 'Blip', 
	 4: 'Chirp', 
	 5: 'Extremely_Loud', 
	 6: 'Helix', 
	 7: 'Koi_Fish', 
	 8: 'Light_Modulation', 
	 9: 'Low_Frequency_Burst', 
	 10: 'Low_Frequency_Lines', 
	 11: 'No_Glitch', 
	 12: 'None_of_the_Above', 
	 13: 'Paired_Doves', 
	 14: 'Power_Line', 
	 15: 'Repeating_Blips', 
	 16: 'Scattered_Light', 
	 17: 'Scratchy', 
	 18: 'Tomte', 
	 19: 'Violin_Mode', 
	 20: 'Wandering_Line', 
	 21: 'Whistle'}

model = load_model('model.h5')

#model.make_predict_function()

def predict_label(img_path):
	i = cv2.imread(img_path)
	i= cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
	ri=cv2.resize(i, (150,150))
	ri=ri/255
	single= np.expand_dims(ri, axis=0)
	pred_single= model.predict(single)
	pred_single1=np.argmax(pred_single, axis=-1)
	print(img_path)
	return dic[pred_single1[0]]

static_dir = os.path.join(os.path.dirname(__file__), 'static')
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Error"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		#img_path = img.filename	
		img.save(img_path)
		#

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
