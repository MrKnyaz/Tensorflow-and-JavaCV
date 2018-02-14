import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Classificator {

    Graph modelGraph;
    Session session;
    List<String> labels;

    public Classificator() {
        try {
            Path modelPath = Paths.get(this.getClass().getResource("trained_model/tensorflow_inception_graph.pb").toURI());
            Path labelsPath = Paths.get(Main.class.getResource("trained_model/imagenet_comp_graph_label_strings.txt").toURI());
            byte[] graphData = Files.readAllBytes(modelPath);
            labels = Files.readAllLines(labelsPath);

            modelGraph = new Graph();
            modelGraph.importGraphDef(graphData);
            session = new Session(modelGraph);

            //Just print two main operations to look at shapes
            System.out.println(modelGraph.operation("input").output(0));
            System.out.println(modelGraph.operation("output").output(0));
        } catch(Exception e) {e.printStackTrace(); throw new RuntimeException(e);}
    }

    public String classify(float[][][][] imageData) {
        Tensor imageTensor = Tensor.create(imageData, Float.class);
        float[][] output = predict(imageTensor);
        return findPredictedLabel(output);
    }

    private float[][] predict(Tensor imageTensor) {
        Tensor result = session.runner()
                .feed("input", imageTensor)
                .fetch("output").run().get(0);
        //create prediction buffer
        float[][] prediction = new float[1][1008];
        result.copyTo(prediction);
        return prediction;
    }

    private String findPredictedLabel(float[][] prediction) {
        int maxValueIndex = 0;
        for (int i = 1; i < prediction[0].length; i++) {
            if (prediction[0][maxValueIndex] < prediction[0][i]) {
                maxValueIndex = i;
            }
        }
        System.out.println(prediction[0][maxValueIndex]);
        return labels.get(maxValueIndex);
    }
}
