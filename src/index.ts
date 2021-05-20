// https://heartbeat.fritz.ai/getting-started-with-image-segmentation-using-tensorflow-js-a157e1cbbd51
const tf = require('@tensorflow/tfjs-node');
import * as bodyPix from '@tensorflow-models/body-pix';
import Jimp from 'jimp';
import {
    BodyPixArchitecture,
    BodyPixOutputStride,
    BodyPixQuantBytes
} from '@tensorflow-models/body-pix/dist/types';
import fs from 'fs';

class RemoveBackground {

    bodymodel!: bodyPix.BodyPix;
    async loadModel() {
        if (!this.bodymodel) {
            const resNet = {
                architecture: "ResNet50" as BodyPixArchitecture,
                outputStride: 16 as BodyPixOutputStride,
                quantBytes: 4 as BodyPixQuantBytes
            };
            this.bodymodel = await bodyPix.load(resNet);
        }
    }
    async Prediction(image: any) {
        await this.loadModel();
        return this.bodymodel.segmentPersonParts(image);
    }
    async removeBG(img: Buffer, output: any) {
        const tfimg = tf.node.decodeImage(img);
        const bodySeg = await this.Prediction(tfimg);
        const jimp = await Jimp.read(img);
        let count = 0;
        for (let i = 0; i < bodySeg.height; i++) {
            for (let j = 0; j < bodySeg.width; j++) {
                if (bodySeg.data[count] === -1) {
                    jimp.setPixelColor(0x00000000, j, i);
                }
                count++;
            }
        }

        await jimp.writeAsync(output);
    }


}


(async () => {
    for(let i = 1; i < 5; i ++){
        const input = `person${i}.jpeg`;
        const output = `output${i}.png`;
        const removeBg = new RemoveBackground();
        const img = fs.readFileSync(input);
        let res = await removeBg.removeBG(img, output);
        console.log('Finished', output)
    }
})();
