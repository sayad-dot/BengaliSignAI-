module com.iut.frontend {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.iut.frontend to javafx.fxml;
    exports com.iut.frontend;
}