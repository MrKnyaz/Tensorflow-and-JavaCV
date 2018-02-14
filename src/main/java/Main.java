import org.tensorflow.*;

import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.Iterator;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        /* ImageProcessor returns 4-dimensional float array -> float[1][224][224][3]
         * First dimension of the array is a batch for neural network,
         * for this simple example I use just one image at a time for classification
         * The code can be easily modified to load and classify multiple images.
         * */
        ImageProcessor imageProcessor = new ImageProcessor();
        Classificator classificator = new Classificator();
        System.out.println(classificator.classify(imageProcessor.loadAndNormalizeImage("images/hyndai.jpg")));
    }
}
