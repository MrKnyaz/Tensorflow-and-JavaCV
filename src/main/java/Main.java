import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        /* ImageProcessor returns 4-dimensional float array -> float[?][224][224][3]
         * First dimension of the array is a batch for neural network,
         * You may pass multiple image paths to ImageProcessor
         * */
        ImageProcessor imageProcessor = new ImageProcessor();
        Classificator classificator = new Classificator();
        float[][][][] imageData = imageProcessor.loadAndNormalizeImages("images/ship.jpg", "images/dog.jpg");
        List<String> result = classificator.classify(imageData);
        for(String label: result) {
            System.out.println(label);
        }
    }
}
