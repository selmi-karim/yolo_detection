
import * as tf from '@tensorflow/tfjs';
import yolo, { downloadModel } from 'tfjs-yolo-tiny';


import { Webcam } from './webcam';

const webcam = new Webcam(document.getElementById('webcam'));
load()

async function load() {
    alert('load start')
    try {
      const model = await tf.loadModel(
      'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
      alert(model)	 
      return model;
    } catch (err) {
      console.error(err);
    }
}

(async function main() {
  try {
    ga();

    alert("Just a heads up! We'll ask to access your webcam so that we can " +
      "detect objects in semi-real-time. \n\nDon't worry, we aren't sending " +
      "any of your images to a remote server, all the ML is being done " +
      "locally on device, and you can check out our source code on Github.");
	
    await webcam.setup();

    doneLoading();
    run();
  } catch(e) {
    console.error(e);
    showError();
  }
})();

async function run() {
  while (true) {
    clearRects();

    const inputImage = webcam.capture();

    const t0 = performance.now();

    const boxes = await yolo(inputImage, model);

    const t1 = performance.now();
    console.log("YOLO inference took " + (t1 - t0) + " milliseconds.");

    boxes.forEach(box => {
      const {
        top, left, bottom, right, classProb, className,
      } = box;

      drawRect(left, top, right-left, bottom-top,
        `${className} Confidence: ${Math.round(classProb * 100)}%`)
    });

    await tf.nextFrame();
  }
}

const webcamElem = document.getElementById('webcam-wrapper');

function drawRect(x, y, w, h, text = '', color = 'red') {
  const rect = document.createElement('div');
  rect.classList.add('rect');
  rect.style.cssText = `top: ${y}; left: ${x}; width: ${w}; height: ${h}; border-color: ${color}`;

  const label = document.createElement('div');
  label.classList.add('label');
  label.innerText = text;
  rect.appendChild(label);

  writeData(text);
  if (text.split(" ")[0] !== "person"){
    var res = "";
    for(var i=0;i<text.split(' ').length-2;i++){
      res+=text.split(' ')[i]+" ";
    };
  }

  webcamElem.appendChild(rect);
}

function clearRects() {
  const rects = document.getElementsByClassName('rect');
  while(rects[0]) {
    rects[0].parentNode.removeChild(rects[0]);
  }
}

function doneLoading() {
  const elem = document.getElementById('loading-message');
  elem.style.display = 'none';

  const successElem = document.getElementById('success-message');
  successElem.style.display = 'block';

  const webcamElem = document.getElementById('webcam-wrapper');
  webcamElem.style.display = 'flex';
}

function showError() {
  const elem = document.getElementById('error-message');
  elem.style.display = 'block';
  doneLoading();
}

function ga() {
  if (process.env.UA) {
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', process.env.UA);
  }
}
