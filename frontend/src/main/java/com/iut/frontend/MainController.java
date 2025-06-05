package com.iut.frontend;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class MainController {

    @FXML private Button selectButton;
    @FXML private Label selectedFileLabel;
    @FXML private Button inferButton;
    @FXML private TextArea resultArea;

    private File selectedFile = null;

    @FXML
    public void initialize() {
        // At startup, Run Inference button is disabled until a file is chosen.
        inferButton.setDisable(true);
    }

    @FXML
    protected void onSelectVideo() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Choose a video file");
        chooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Video Files", "*.mp4", "*.avi")
        );
        File file = chooser.showOpenDialog(selectButton.getScene().getWindow());
        if (file != null) {
            selectedFile = file;
            selectedFileLabel.setText(file.getAbsolutePath());
            inferButton.setDisable(false);
            resultArea.clear();
        }
    }

    @FXML
    protected void onRunInference() {
        if (selectedFile == null) {
            showAlert("No file chosen", "Please select a video file first.");
            return;
        }

        // Placeholder: show “Running inference…” then a dummy result.
        resultArea.setText("Running inference on:\n" + selectedFile.getName());

        // Simulate a delay, then display dummy prediction
        new Thread(() -> {
            try {
                Thread.sleep(1000); // simulate processing
            } catch (InterruptedException e) {
                // ignore
            }
            Platform.runLater(() -> resultArea.setText("Predicted: উদাহরণ-লেবেল")); // “example-label” in Bangla
        }).start();

        // **Later**, replace above with code to start your Python script or ONNX model,
        // then read its output and set resultArea.setText(predictionString).
    }

    private void showAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.WARNING);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
