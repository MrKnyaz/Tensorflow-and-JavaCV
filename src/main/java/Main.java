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
        ImageProcessor imageProcessor = new ImageProcessor();
        Classificator classificator = new Classificator();
        System.out.println(classificator.classify(imageProcessor.loadAndNormalizeImage("images/hyndai.jpg")));
    }
}
