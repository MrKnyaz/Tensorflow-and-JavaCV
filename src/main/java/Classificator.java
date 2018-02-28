import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Classificator {

    private Graph modelGraph;
    private Session session;
    private List<String> labels;

    public Classificator() {
        try {
            Path modelPath = Paths.get(this.getClass().getResource("trained_model/tensorflow_inception_graph.pb").toURI());
            Path labelsPath = Paths.get(Main.class.getResource("trained_model/imagenet_comp_graph_label_strings.txt").toURI());
            byte[] graphData = Files.readAllBytes(modelPath);
            labels = Files.readAllLines(labelsPath);

            modelGraph = new Graph();
            modelGraph.importGraphDef(graphData);
            session = new Session(modelGraph);
        } catch(Exception e) {e.printStackTrace(); throw new RuntimeException(e);}
    }

    public List<String> classify(float[][][][] imageData) {
        Tensor imageTensor = Tensor.create(imageData, Float.class);
        float[][] prediction = predict(imageTensor);
        return findPredictedLabel(prediction);
    }

    private float[][] predict(Tensor imageTensor) {
        Tensor result = session.runner()
                .feed("input", imageTensor)
                .fetch("output").run().get(0);
        int batchSize = (int)result.shape()[0];
        //create prediction buffer
        float[][] prediction = new float[batchSize][1008];
        result.copyTo(prediction);
        return prediction;
    }

    private List<String> findPredictedLabel(float[][] prediction) {
        List<String> result = new ArrayList<>();
        int batchSize = prediction.length;
        for (int i = 0; i < batchSize; i++) {
            //Finding maximum value for each predicted image
            int maxValueIndex = 0;
            for (int j = 1; j < prediction[i].length; j++) {
                if (prediction[i][maxValueIndex] < prediction[i][j]) {
                    maxValueIndex = j;
                }
            }
            result.add(labels.get(maxValueIndex) + ": " + (prediction[i][maxValueIndex] * 100) + "%");
        }
        return result;
    }
}
