import org.bytedeco.javacv.*;

import java.awt.image.BufferedImage;
import java.io.File;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvSaveImage;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * Image Processor for Model https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
 */
public class ImageProcessor {
    /* Constants specific to trained model
    * All RGB values are converted to normalized float values
    * using this formula (value - mean) / scale
    * */
    private final int height = 224;
    private final int width = 224;
    private final float mean = 117f;
    private final float scale = 1f;


    public ImageProcessor() {
    }

    public float[][][][] loadAndNormalizeImage(String path) {
        IplImage origImg = cvLoadImage(getFullPath(path));
        //Creating image placeholder to put resized image data
        IplImage resizedImg = IplImage.create(width, height, origImg.depth(), origImg.nChannels());
        cvResize(origImg, resizedImg);
        return getRGBArray(resizedImg);
    }

    private float[][][][] getRGBArray(IplImage image) {
        float[][][][] result = new float[1][image.height()][image.width()][3];
        for (int i = 0; i < image.height(); i++) {
            for (int j = 0; j < image.width(); j++) {
                CvScalar pixel = cvGet2D(image, i, j);
                result[0][i][j][0] = (float)(pixel.val(2) - mean) / scale; //R
                result[0][i][j][1] = (float)(pixel.val(1) - mean) / scale; //G
                result[0][i][j][2] = (float)(pixel.val(0) - mean) / scale; //B
            }
        }
        return result;
    }

    private String getFullPath(String path) {
        return new File(this.getClass().getResource(path).getFile()).getAbsolutePath();
    }
}
