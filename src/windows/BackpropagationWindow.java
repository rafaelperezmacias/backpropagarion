package windows;

import models.Backpropagation;
import utils.DataSet;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.ArrayList;
import java.util.Arrays;

public class BackpropagationWindow extends JFrame {

    private ArrayList<Point> points;

    private final int MAP_WIDTH = 500;
    private final int MAP_HEIGHT = 500;
    private final int RADIUS_POINT = 5;
    private final double MAP_SCALE = 5.0;

    private final Map map;

    private ArrayList<Color> targetColors;
    private ArrayList<String> targets;
    private int idxTarget;

    private final JMenu jmOptions;
    private final JMenu jmPredict;
    private final JMenu jmPerceptron;
    private final JRadioButtonMenuItem jmiPerceptronPredict;

    private final JRadioButton jrbFirstHiddenLayer;
    private final JRadioButton jrbSecondHiddenLayer;
    private final JRadioButton rbtnBatch;
    private final JRadioButton rbtnStochastic;
    private final JRadioButton rbtnMiniBatch;

    private final JTextField txtFHL;
    private final JTextField txtSHL;
    private final JTextField txtLearningRate;
    private final JTextField txtEpochs;
    private final JTextField txtMinError;
    private final JTextField txtMiniBatch;

    private final JLabel lblFHL;
    private final JLabel lblSHL;

    private final JButton btnBackpropagation;
    private final JButton btnAddInstance;

    private final JLabel lblLearningRateResult;
    private final JLabel lblEpochResult;
    private final JLabel lblErrorResult;

    private final JComboBox<String> jcbValuesOfInstances;
    private final JComboBox<String> jcbFirstLayerFunction;
    private final JComboBox<String> jcbSecondLayerFunction;
    private final JComboBox<String> jcbLastLayerFunction;

    private boolean clickEnable;
    private boolean addInstanceEnable;
    private boolean modelEnable;

    private BackpropagationThread.Model model;

    private ErrorChartBPWindow errorChartBPWindow;

    public BackpropagationWindow()
    {
        super("Ejemplo de una MLN con Backpropagation");
        setLayout(null);
        setSize(1100,625);
        setLocationRelativeTo(null);
        // Inicializamos la lista que contiene los puntos del mapa
        points = new ArrayList<>();
        // Incializamos la lista con colores
        targetColors = new ArrayList<>();
        targetColors.add(Color.BLUE);
        targetColors.add(Color.GREEN);
        // Inicializamos la lista con los objetivos
        targets = new ArrayList<>();
        targets.add("Azul");
        targets.add("Verde");
        // Barra de menu
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
        // Un menu de la barra
        jmOptions = new JMenu("Opciones");
        menuBar.add(jmOptions);
        // Opciones del menu
        // ELiminar ultima
        JMenuItem jmiDeleteLastInstance = new JMenuItem("Eliminar la ultima instancia");
        jmiDeleteLastInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( !points.isEmpty() ) {
                    int idxPoint = -1;
                    for ( int i = points.size() - 1; i >= 0; i-- ) {
                        if ( !points.get(i).sweep ) {
                            idxPoint = i;
                            break;
                        }
                    }
                    if ( idxPoint != -1 ) {
                        points.remove(idxPoint);
                        map.repaint();
                    }
                }
            }
        });
        jmOptions.add(jmiDeleteLastInstance);
        // Limpiar instancias
        JMenuItem jmiClearInstances = new JMenuItem("Limpiar instancias");
        jmOptions.add(jmiClearInstances);
        // Limpiar barrido
        JMenuItem jmiClearSweep = new JMenuItem("Limpiar barrido");
        jmOptions.add(jmiClearSweep);
        // Limpiar el programa
        JMenuItem jmiClearAll = new JMenuItem("Limpiar todo");
        jmiClearAll.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                points.clear();
                map.repaint();
                targets.clear();
                targetColors.clear();
                targetColors.add(Color.BLUE);
                targetColors.add(Color.GREEN);
                targets.add("Azul");
                targets.add("Verde");
                jcbValuesOfInstances.removeAllItems();
                jcbValuesOfInstances.addItem("0 - Azul");
                jcbValuesOfInstances.addItem("1 - Verde");
                addInstanceEnable = true;
                modelEnable = false;
                clickEnable = true;
                jcbValuesOfInstances.setSelectedIndex(0);
                txtFHL.setText("");
                txtSHL.setText("");
                txtLearningRate.setText("0.");
                txtEpochs.setText("");
                txtMinError.setText("");
                txtMiniBatch.setText("");
                txtMiniBatch.setEnabled(false);
                rbtnStochastic.setSelected(true);
                jrbFirstHiddenLayer.setSelected(false);
                txtFHL.setEnabled(false);
                jrbSecondHiddenLayer.setSelected(false);
                jrbSecondHiddenLayer.setEnabled(false);
                txtSHL.setEnabled(false);
                jcbFirstLayerFunction.setEnabled(false);
                jcbFirstLayerFunction.setSelectedIndex(0);
                jcbSecondLayerFunction.setEnabled(false);
                jcbSecondLayerFunction.setSelectedIndex(0);
                jcbLastLayerFunction.setSelectedIndex(0);
            }
        });
        jmOptions.add(jmiClearAll);
        // Salir
        JMenuItem jmiClose = new JMenuItem("Salir");
        jmiClose.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                BackpropagationWindow.this.dispose();
            }
        });
        jmOptions.add(jmiClose);
        // Opciones para predecir
        jmPredict = new JMenu("Modelo");
        jmPredict.setVisible(false);
        menuBar.add(jmPredict);
        ButtonGroup bgPredict = new ButtonGroup();
        /** Perceptron */
        jmPerceptron = new JMenu("Backpropagation");
        jmPerceptron.setVisible(true);
        jmPredict.add(jmPerceptron);
        // Predecir una instancia
        jmiPerceptronPredict = new JRadioButtonMenuItem("Predecir", true);
        bgPredict.add(jmiPerceptronPredict);
        jmPerceptron.add(jmiPerceptronPredict);
        jmiPerceptronPredict.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( modelEnable ) {
                    addInstanceEnable = false;
                }
            }
        });
        /** Instancia */
        // Agregar una nueva instancia
        JRadioButtonMenuItem jmiNewInstance = new JRadioButtonMenuItem("Nueva instancia");
        bgPredict.add(jmiNewInstance);
        jmPredict.add(jmiNewInstance);
        jmiNewInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                addInstanceEnable = true;
            }
        });
        // Limpiar instancias
        jmiClearInstances.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point point : points ) {
                    if ( point.sweep ) {
                        newPoints.add(point);
                    }
                }
                points = newPoints;
                map.repaint();
                jmiNewInstance.setSelected(true);
                addInstanceEnable = true;
            }
        });
        // Limpiar barrido
        jmiClearSweep.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point point : points ) {
                    if ( !point.sweep ) {
                        newPoints.add(point);
                    }
                }
                points = newPoints;
                map.repaint();
                jmiNewInstance.setSelected(true);
                addInstanceEnable = true;
            }
        });
        // Lienzo princiapal de la ventana
        map = new Map();
        map.setSize(MAP_WIDTH, MAP_HEIGHT);
        map.setLocation(35,30);
        map.setBackground(Color.WHITE);
        add(map);
        // Bandera para validaciones / acciones
        clickEnable = true;
        modelEnable = false;
        addInstanceEnable = true;
        // Eventos del mouse
        map.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                Point point = new Point();
                // Tratamiento de los datos
                point.xMap = e.getX();
                point.yMap = e.getY();
                point.x = ( e.getX() >= MAP_WIDTH * 0.5 ) ? e.getX() - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - e.getX());
                point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                point.y = ( e.getY() > MAP_HEIGHT * 0.5 ) ? -(e.getY() - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - e.getY();
                point.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                // Predecir
                if ( !addInstanceEnable && modelEnable && e.getButton() == MouseEvent.BUTTON1 ) {
                    Object[] instance = new Object[2];
                    instance[0] = point.x;
                    instance[1] = point.y;
                    Object[] result;
                    try {
                        result = model.predictWithPercentage(instance);
                        point.sweep = false;
                        point.target = Integer.parseInt((String) result[0]);
                        point.color = targetColors.get(point.target);
                        JOptionPane.showMessageDialog(null, "La nueva instancia es: " + targets.get(point.target), "Alerta", JOptionPane.INFORMATION_MESSAGE);
                        points.add(point);
                        map.repaint();
                    } catch (Exception ex) {
                        ex.printStackTrace();
                        System.out.println("No se pudo realizar el barrido");
                    }
                    return;
                }
                // Boton izquierdo
                if ( e.getButton() == MouseEvent.BUTTON1 && clickEnable ) {
                    point.sweep = false;
                    point.target = idxTarget;
                    point.color = targetColors.get(point.target);
                    points.add(point);
                    System.out.println("Nuevo punto agregado: " + point);
                    map.repaint();
                    modelEnable = false;
                    jmPredict.setVisible(false);
                }
            }
        });
        /** Titulos, leyendas */
        // Escala del norte del plano
        JLabel lblScaleNorth = new JLabel("+ " + MAP_SCALE);
        lblScaleNorth.setSize(28,10);
        lblScaleNorth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() - 15);
        add(lblScaleNorth);
        // Escala del sur del plano
        JLabel lblScaleSouth = new JLabel("- " + MAP_SCALE);
        lblScaleSouth.setSize(28,10);
        lblScaleSouth.setLocation(map.getX() + ( map.getWidth() / 2 ) - 12, map.getY() + map.getHeight() + 5);
        add(lblScaleSouth);
        // Escala del este del plano
        JLabel lblScaleEast = new JLabel("+ " + MAP_SCALE);
        lblScaleEast.setSize(28,10);
        lblScaleEast.setLocation(map.getX() + ( map.getWidth() ) + 5, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleEast);
        // Escala del este del plano
        JLabel lblScaleWest = new JLabel("- " + MAP_SCALE);
        lblScaleWest.setSize(28,10);
        lblScaleWest.setLocation(map.getX() - 27, map.getY() + (map.getHeight() / 2) - 5);
        add(lblScaleWest);
        // Configuracion de las instancias
        JLabel lblTitleInstances = new JLabel("Configuracion de los instancias");
        lblTitleInstances.setLocation(map.getX() + map.getWidth() + 40, map.getY());
        lblTitleInstances.setSize(getWidth() - (map.getX() + map.getWidth() + 75), 24);
        lblTitleInstances.setHorizontalAlignment(JLabel.CENTER);
        lblTitleInstances.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblTitleInstances);
        // Subtitulo
        JLabel lblInstance = new JLabel("Instancia actual: ");
        lblInstance.setLocation(lblTitleInstances.getX(), lblTitleInstances.getY() + lblTitleInstances.getHeight() + 10);
        lblInstance.setSize((int) (lblTitleInstances.getWidth() * 0.28), 30);
        lblInstance.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblInstance);
        // Instancias
        jcbValuesOfInstances = new JComboBox<>();
        jcbValuesOfInstances.setLocation(lblInstance.getX() + lblInstance.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), lblInstance.getY());
        jcbValuesOfInstances.setSize((int) (lblTitleInstances.getWidth() * 0.58), 30);
        jcbValuesOfInstances.addItem("0 - Azul");
        jcbValuesOfInstances.addItem("1 - Verde");
        jcbValuesOfInstances.addItemListener(new ItemListener() {
            @Override
            public void itemStateChanged(ItemEvent e) {
                idxTarget = jcbValuesOfInstances.getSelectedIndex();
            }
        });
        add(jcbValuesOfInstances);
        // Boton para agregar mas instancias
        btnAddInstance = new JButton("+");
        btnAddInstance.setLocation(jcbValuesOfInstances.getX() + jcbValuesOfInstances.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), lblInstance.getY());
        btnAddInstance.setSize((int) (lblTitleInstances.getWidth() * 0.10), 30);
        btnAddInstance.setBackground(new Color(71, 138, 201));
        btnAddInstance.setOpaque(true);
        btnAddInstance.setForeground(Color.WHITE);
        btnAddInstance.setFont(new Font("Dialog", Font.BOLD, 16));
        btnAddInstance.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnAddInstance.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                Color newColor = JColorChooser.showDialog(null, "Seleccione un color", Color.RED);
                String newName = JOptionPane.showInputDialog(null, "Nombre: ", "Color del nuevo objetivo", JOptionPane.QUESTION_MESSAGE);
                if ( newColor == null || newName == null || newName.isEmpty() ) {
                    JOptionPane.showMessageDialog(null, "No se pudo crear el nuevo objetivo", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                jcbValuesOfInstances.addItem(targetColors.size() + " - " + newName);
                targets.add(newName);
                targetColors.add(newColor);
                jcbValuesOfInstances.setSelectedIndex(targetColors.size() - 1);
                JOptionPane.showMessageDialog(null, "Nuevo objetivo creado con exito", "Resultado", JOptionPane.INFORMATION_MESSAGE);
            }
        });
        add(btnAddInstance);
        // Separador
        JSeparator jsTargets = new JSeparator();
        jsTargets.setOrientation(SwingConstants.HORIZONTAL);
        jsTargets.setSize(lblTitleInstances.getWidth(), 2);
        jsTargets.setLocation(lblTitleInstances.getX(), btnAddInstance.getY() + btnAddInstance.getHeight() + 10);
        add(jsTargets);
        // Configuracion de la arquitectura
        JLabel lblArchitecture = new JLabel("Configuracion de la arquitectura");
        lblArchitecture.setLocation(map.getX() + map.getWidth() + 40, jsTargets.getY() + 6);
        lblArchitecture.setSize(getWidth() - (map.getX() + map.getWidth() + 75), 24);
        lblArchitecture.setHorizontalAlignment(JLabel.CENTER);
        lblArchitecture.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblArchitecture);
        // Primera capa oculta
        jrbFirstHiddenLayer = new JRadioButton("1ra capa oculta");
        jrbFirstHiddenLayer.setLocation(lblTitleInstances.getX(), lblArchitecture.getY() + lblArchitecture.getHeight() + 10);
        jrbFirstHiddenLayer.setSize((int) (lblTitleInstances.getWidth() * 0.28), 30);
        jrbFirstHiddenLayer.setFont(new Font("Dialog", Font.PLAIN, 14));
        jrbFirstHiddenLayer.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( jrbFirstHiddenLayer.isSelected() ) {
                    txtFHL.setEnabled(true);
                    lblFHL.setEnabled(true);
                    jcbFirstLayerFunction.setEnabled(true);
                    jrbSecondHiddenLayer.setEnabled(true);
                } else {
                    jcbFirstLayerFunction.setEnabled(false);
                    jcbSecondLayerFunction.setEnabled(false);
                    txtFHL.setEnabled(false);
                    lblFHL.setEnabled(false);
                    jrbSecondHiddenLayer.setEnabled(false);
                    txtSHL.setEnabled(false);
                    lblSHL.setEnabled(false);
                    jrbSecondHiddenLayer.setSelected(false);
                }
            }
        });
        add(jrbFirstHiddenLayer);
        // Leyenda
        lblFHL = new JLabel("Neuronas: ");
        lblFHL.setLocation(jrbFirstHiddenLayer.getX() + jrbFirstHiddenLayer.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbFirstHiddenLayer.getY());
        lblFHL.setSize((int) (lblTitleInstances.getWidth() * 0.18), 30);
        lblFHL.setFont(new Font("Dialog", Font.PLAIN, 14));
        lblFHL.setEnabled(false);
        add(lblFHL);
        // Campo
        txtFHL = new JTextField();
        txtFHL.setLocation(lblFHL.getX() + lblFHL.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbFirstHiddenLayer.getY());
        txtFHL.setSize((int) (lblTitleInstances.getWidth() * 0.25), 30);
        txtFHL.setEnabled(false);
        txtFHL.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtFHL);
        // Funcion de activacion
        jcbFirstLayerFunction = new JComboBox<>();
        jcbFirstLayerFunction.setLocation(txtFHL.getX() + txtFHL.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbFirstHiddenLayer.getY());
        jcbFirstLayerFunction.setSize((int) (lblTitleInstances.getWidth() * 0.23), 30);
        jcbFirstLayerFunction.addItem("Sigmoide");
        jcbFirstLayerFunction.addItem("Tanh");
        jcbFirstLayerFunction.addItem("ReLU");
        jcbFirstLayerFunction.setEnabled(false);
        add(jcbFirstLayerFunction);
        //Segunda capa oculta
        jrbSecondHiddenLayer = new JRadioButton("2da capa oculta");
        jrbSecondHiddenLayer.setLocation(lblTitleInstances.getX(), jrbFirstHiddenLayer.getY() + jrbFirstHiddenLayer.getHeight() + 10);
        jrbSecondHiddenLayer.setSize((int) (lblTitleInstances.getWidth() * 0.28), 30);
        jrbSecondHiddenLayer.setFont(new Font("Dialog", Font.PLAIN, 14));
        jrbSecondHiddenLayer.setEnabled(false);
        jrbSecondHiddenLayer.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( jrbSecondHiddenLayer.isSelected() ) {
                    txtSHL.setEnabled(true);
                    lblSHL.setEnabled(true);
                    jcbSecondLayerFunction.setEnabled(true);
                } else {
                    txtSHL.setEnabled(false);
                    lblSHL.setEnabled(false);
                    jcbSecondLayerFunction.setEnabled(false);
                }
            }
        });
        add(jrbSecondHiddenLayer);
        // Leyenda
        lblSHL = new JLabel("Neuronas: ");
        lblSHL.setLocation(jrbSecondHiddenLayer.getX() + jrbSecondHiddenLayer.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbSecondHiddenLayer.getY());
        lblSHL.setSize((int) (lblTitleInstances.getWidth() * 0.18), 30);
        lblSHL.setFont(new Font("Dialog", Font.PLAIN, 14));
        lblSHL.setEnabled(false);
        add(lblSHL);
        // Campo
        txtSHL = new JTextField();
        txtSHL.setLocation(lblSHL.getX() + lblSHL.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbSecondHiddenLayer.getY());
        txtSHL.setSize((int) (lblTitleInstances.getWidth() * 0.25), 30);
        txtSHL.setEnabled(false);
        txtSHL.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtSHL);
        // Funcion de activacion
        jcbSecondLayerFunction = new JComboBox<>();
        jcbSecondLayerFunction.setLocation(txtSHL.getX() + txtSHL.getWidth() + ((int) (lblTitleInstances.getWidth() * 0.02)), jrbSecondHiddenLayer.getY());
        jcbSecondLayerFunction.setSize((int) (lblTitleInstances.getWidth() * 0.23), 30);
        jcbSecondLayerFunction.addItem("Sigmoide");
        jcbSecondLayerFunction.addItem("Tanh");
        jcbSecondLayerFunction.addItem("ReLU");
        jcbSecondLayerFunction.setEnabled(false);
        add(jcbSecondLayerFunction);
        // Capa de salida
        JLabel lblLastLayer = new JLabel("Capa de salida");
        lblLastLayer.setLocation(jrbFirstHiddenLayer.getX(), jrbSecondHiddenLayer.getY() + jrbSecondHiddenLayer.getHeight() + 10);
        lblLastLayer.setSize((int) (lblTitleInstances.getWidth() * 0.23),30);
        lblLastLayer.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblLastLayer);
        // Funcion de activacion
        jcbLastLayerFunction = new JComboBox<>();
        jcbLastLayerFunction.setSize((int) (lblTitleInstances.getWidth() * 0.75), 30);
        jcbLastLayerFunction.setLocation(lblLastLayer.getX() + lblLastLayer.getWidth() + (int) (lblTitleInstances.getWidth() * 0.02), lblLastLayer.getY());
        jcbLastLayerFunction.addItem("Sigmoide");
        jcbLastLayerFunction.addItem("Tanh");
        jcbLastLayerFunction.addItem("ReLU");
        add(jcbLastLayerFunction);
        // Separador
        JSeparator jsArchitecture = new JSeparator();
        jsArchitecture.setOrientation(SwingConstants.HORIZONTAL);
        jsArchitecture.setSize(lblTitleInstances.getWidth(), 2);
        jsArchitecture.setLocation(lblTitleInstances.getX(), txtSHL.getY() + txtSHL.getHeight() + 50);
        add(jsArchitecture);
        // Configuracion de los parametros
        JLabel lblParameters = new JLabel("Hiperparametros");
        lblParameters.setLocation(map.getX() + map.getWidth() + 40, jsArchitecture.getY() + 6);
        lblParameters.setSize((int) (lblTitleInstances.getWidth() * 0.49), 24);
        lblParameters.setHorizontalAlignment(JLabel.CENTER);
        lblParameters.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblParameters);
        // Factor de aprendizaje
        JLabel lblLearningRate = new JLabel("Learning rate: ");
        lblLearningRate.setLocation(lblParameters.getX(), lblParameters.getY() + lblParameters.getHeight() + 10);
        lblLearningRate.setSize((int) (lblParameters.getWidth() * 0.40), 30);
        lblLearningRate.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblLearningRate);
        txtLearningRate = new JTextField("0.");
        txtLearningRate.setLocation(lblLearningRate.getX() + lblLearningRate.getWidth() + ((int) (lblParameters.getWidth() * 0.02)), lblLearningRate.getY());
        txtLearningRate.setSize((int) (lblParameters.getWidth() * 0.58), 30);
        txtLearningRate.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' || txtLearningRate.getCaretPosition() < 2 ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
            @Override
            public void keyPressed(KeyEvent e) {
                if ( txtLearningRate.getText().length() == 2 && e.getKeyCode() == KeyEvent.VK_BACK_SPACE ) {
                    e.consume();
                }
                super.keyPressed(e);
            }
        });
        add(txtLearningRate);
        // Epocas
        JLabel lblEpochs = new JLabel("Epocas: ");
        lblEpochs.setSize(lblLearningRate.getSize());
        lblEpochs.setLocation(lblLearningRate.getX(), lblLearningRate.getY() + lblLearningRate.getHeight() + 10);
        lblEpochs.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblEpochs);
        txtEpochs = new JTextField();
        txtEpochs.setSize(txtLearningRate.getSize());
        txtEpochs.setLocation(txtLearningRate.getX(), lblEpochs.getY());
        txtEpochs.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        add(txtEpochs);
        // Error minimo
        JLabel lblError = new JLabel("Error minimo: ");
        lblError.setLocation(lblParameters.getX(), lblEpochs.getY() + lblEpochs.getHeight() + 10);
        lblError.setSize(lblLearningRate.getSize());
        lblError.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblError);
        txtMinError = new JTextField();
        txtMinError.setLocation(lblError.getX() + lblError.getWidth() + ((int) (lblParameters.getWidth() * 0.02)), lblError.getY());
        txtMinError.setSize(txtLearningRate.getSize());
        txtMinError.addKeyListener(new CustomKeyListener(txtMinError));
        add(txtMinError);
        // Tipo de gradiente
        // Separador
        JSeparator jsGradient = new JSeparator();
        jsGradient.setOrientation(SwingConstants.VERTICAL);
        jsGradient.setSize(2, 150);
        jsGradient.setLocation(map.getX() + map.getWidth() + 40 + (int) (lblTitleInstances.getWidth() * 0.5), lblParameters.getY());
        add(jsGradient);
        // Configuracion del tipo de gradiente
        JLabel lblGradient = new JLabel("Tipo de gradiente");
        lblGradient.setLocation(map.getX() + map.getWidth() + 40 + (int) (lblTitleInstances.getWidth() * 0.51), jsGradient.getY());
        lblGradient.setSize((int) (lblTitleInstances.getWidth() * 0.49), 24);
        lblGradient.setHorizontalAlignment(JLabel.CENTER);
        lblGradient.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblGradient);
        // Agrupador de botones
        ButtonGroup btnGradientGroup = new ButtonGroup();
        // Estocastico
        rbtnStochastic = new JRadioButton("Estocastico");
        rbtnStochastic.setSize(lblLearningRate.getWidth(), 30);
        rbtnStochastic.setLocation(lblGradient.getX() + 20, lblLearningRate.getY());
        rbtnStochastic.setSelected(true);
        rbtnStochastic.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( rbtnStochastic.isSelected() ) {
                    txtMiniBatch.setEnabled(false);
                }
            }
        });
        btnGradientGroup.add(rbtnStochastic);
        add(rbtnStochastic);
        // Batch
        rbtnBatch = new JRadioButton("Batch");
        rbtnBatch.setSize(lblLearningRate.getWidth(), 30);
        rbtnBatch.setLocation(lblGradient.getX() + 20, lblEpochs.getY());
        rbtnBatch.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( rbtnBatch.isSelected() ) {
                    txtMiniBatch.setEnabled(false);
                }
            }
        });
        btnGradientGroup.add(rbtnBatch);
        add(rbtnBatch);
        // Mini batch
        rbtnMiniBatch = new JRadioButton("Mini-batch");
        rbtnMiniBatch.setSize((int) (lblGradient.getWidth() * 0.49), 30);
        rbtnMiniBatch.setLocation(lblGradient.getX() + 20, lblError.getY());
        btnGradientGroup.add(rbtnMiniBatch);
        add(rbtnMiniBatch);
        // Input para los batch
        txtMiniBatch = new JTextField();
        txtMiniBatch.setSize((int) (lblGradient.getWidth() * 0.49), rbtnMiniBatch.getHeight());
        txtMiniBatch.setLocation(rbtnMiniBatch.getX() + rbtnMiniBatch.getWidth() + (int) (lblGradient.getWidth() * 0.02) - 20, rbtnMiniBatch.getY());
        txtMiniBatch.setEnabled(false);
        txtMiniBatch.addKeyListener(new KeyAdapter() {
            @Override
            public void keyTyped(KeyEvent e) {
                if ( e.getKeyChar() < '0' || e.getKeyChar() > '9' ) {
                    e.consume();
                }
                super.keyTyped(e);
            }
        });
        rbtnMiniBatch.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                if ( rbtnMiniBatch.isSelected() ) {
                    txtMiniBatch.setEnabled(true);
                }
            }
        });
        add(txtMiniBatch);
        // Inicio del algoritmo
        btnBackpropagation = new JButton("Backpropagation");
        btnBackpropagation.setSize(lblTitleInstances.getWidth(), 40);
        btnBackpropagation.setLocation(lblParameters.getX(), txtMinError.getY() + txtMinError.getHeight() + 10);
        btnBackpropagation.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btnBackpropagation.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                double learningRate;
                int epochs;
                double minError;
                int neuronsFHL = 0;
                int neuronsSHL = 0;
                int batchSize = 0;
                try {
                    learningRate = Double.parseDouble(txtLearningRate.getText());
                    epochs = Integer.parseInt(txtEpochs.getText());
                    minError = Double.parseDouble(txtMinError.getText());
                    if ( jrbFirstHiddenLayer.isSelected() ) {
                        neuronsFHL = Integer.parseInt(txtFHL.getText());
                    }
                    if ( jrbSecondHiddenLayer.isSelected() ) {
                        neuronsSHL = Integer.parseInt(txtSHL.getText());
                    }
                    if ( rbtnMiniBatch.isSelected() ) {
                        batchSize = Integer.parseInt(txtMiniBatch.getText());
                    }
                } catch (NumberFormatException ex) {
                    JOptionPane.showMessageDialog(null, "Parametros no especificados o incorrectos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( epochs <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las epocas no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( learningRate == 0 ) {
                    JOptionPane.showMessageDialog(null, "El factor de aprendizaje no puede ser 0", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( minError < 0 ) {
                    JOptionPane.showMessageDialog(null, "El error no puede ser negativo", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( jrbFirstHiddenLayer.isSelected() && neuronsFHL <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las neuronas de la capa 1 no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( jrbSecondHiddenLayer.isSelected() && neuronsSHL <= 0 ) {
                    JOptionPane.showMessageDialog(null, "Las neuronas de la capa 2 no pueden ser 0 o menos", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                if ( points.size() < 1 ) {
                    JOptionPane.showMessageDialog(null, "Ingrese minimo una instancia", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                // Eliminamos los puntos de barrido
                ArrayList<Point> newPoints = new ArrayList<>();
                for ( Point tmpPoint : points ) {
                    if ( !tmpPoint.sweep ) {
                        newPoints.add(tmpPoint);
                    }
                }
                points = newPoints;
                map.repaint();
                // Validacion del batch
                if ( rbtnMiniBatch.isSelected() && (batchSize < 1 || batchSize > points.size()) ) {
                    JOptionPane.showMessageDialog(null, "El mini-batch esta fuera de rango (0 - " + points.size() + ")", "Error", JOptionPane.ERROR_MESSAGE);
                    return;
                }
                // Deshabilitamos la interfaz temporalmente y otras cosas
                changeUIState(false);
                jmPredict.setVisible(false);
                clickEnable = false;
                modelEnable = false;
                // Creacion del conjunto de datos
                String[] headers = { "x_1", "x_2", "y" };
                String[] attributeTypes = { DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE, DataSet.NUMERIC_TYPE };
                DataSet dataSet;
                try {
                    dataSet = DataSet.getEmptyDataSetWithHeaders(headers, attributeTypes, "y");
                } catch (Exception ex) {
                    System.out.println("El dataset no pudo ser creado");
                    return;
                }
                for ( Point point : points ) {
                    try {
                        dataSet.addInstance(new ArrayList<>(Arrays.asList("" + point.x,"" + point.y, "" + point.target)));
                    } catch (Exception ex) {
                        System.out.println("No se pudo agregar la instancia del punto " + point);
                    }
                }
                System.out.println("Conjunto de datos con el que el algoritmo trabajara");
                System.out.println(dataSet);
                // Parametros del algoritmo
                BackpropagationThread.Params params = new BackpropagationThread.Params();
                ArrayList<BackpropagationThread.Layer> hiddenLayers = new ArrayList<>();
                if ( jrbFirstHiddenLayer.isSelected() ) {
                    hiddenLayers.add(new BackpropagationThread.Layer(neuronsFHL, getFunction(jcbFirstLayerFunction.getSelectedIndex())));
                }
                if ( jrbSecondHiddenLayer.isSelected() ) {
                    hiddenLayers.add(new BackpropagationThread.Layer(neuronsSHL, getFunction(jcbSecondLayerFunction.getSelectedIndex())));
                }
                params.setLastNeuronFunctionActivation(getFunction(jcbLastLayerFunction.getSelectedIndex()));
                params.setHiddenLayers(hiddenLayers);
                params.setEpochs(epochs);
                params.setMinError(minError);
                params.setLearningRate(learningRate);
                // Tipo de gradiente
                if ( rbtnStochastic.isSelected() ) {
                    params.setGradient(Backpropagation.STOCHASTIC_GRADIENT);
                } else if ( rbtnBatch.isSelected() ) {
                    params.setGradient(Backpropagation.BATCH_GRADIENT);
                } else {
                    params.setGradient(Backpropagation.MINI_BATCH_GRADIENT);
                    params.setBatchSize(batchSize);
                }
                // Obtenemos el modelo
                try {
                    BackpropagationThread backpropagationThread = new BackpropagationThread();
                    backpropagationThread.makeModel(dataSet, params, BackpropagationWindow.this);
                    if ( errorChartBPWindow != null ) {
                        errorChartBPWindow.dispose();
                    }
                    errorChartBPWindow = new ErrorChartBPWindow(BackpropagationWindow.this);
                } catch (Exception ex) {
                    System.out.println("El modelo no se pudo generar");
                    ex.printStackTrace();
                }
            }
        });
        add(btnBackpropagation);
        // Separador
        JSeparator jsResults = new JSeparator();
        jsResults.setOrientation(SwingConstants.HORIZONTAL);
        jsResults.setSize(lblTitleInstances.getWidth(), 2);
        jsResults.setLocation(lblTitleInstances.getX(), btnBackpropagation.getY() + btnBackpropagation.getHeight() + 6);
        add(jsResults);
        // Configuracion de los parametros
        JLabel lblResults = new JLabel("Resultados");
        lblResults.setLocation(map.getX() + map.getWidth() + 40, jsResults.getY() + 6);
        lblResults.setSize(getWidth() - (map.getX() + map.getWidth() + 75), 24);
        lblResults.setHorizontalAlignment(JLabel.CENTER);
        lblResults.setFont(new Font("Dialog", Font.BOLD, 16));
        add(lblResults);
        // Learning rate
        lblLearningRateResult = new JLabel("<html>Learning rate: <b>0.0</b></html>");
        lblLearningRateResult.setLocation(lblTitleInstances.getX(), lblResults.getY() + lblResults.getHeight() + 5);
        lblLearningRateResult.setSize(lblTitleInstances.getWidth(), 18);
        lblLearningRateResult.setHorizontalAlignment(JLabel.LEFT);
        lblLearningRateResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblLearningRateResult);
        // Epoca
        lblEpochResult = new JLabel("<html>Epoca: <b>0</b></html>");
        lblEpochResult.setLocation(lblTitleInstances.getX(), lblLearningRateResult.getY() + lblLearningRateResult.getHeight() + 5);
        lblEpochResult.setSize(lblTitleInstances.getWidth(), 18);
        lblEpochResult.setHorizontalAlignment(JLabel.LEFT);
        lblEpochResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblEpochResult);
        // Error minimo
        lblErrorResult = new JLabel("<html>Error: <b>0.0</b></html>");
        lblErrorResult.setLocation(lblTitleInstances.getX(), lblEpochResult.getY() + lblEpochResult.getHeight() + 5);
        lblErrorResult.setSize(lblTitleInstances.getWidth(), 18);
        lblErrorResult.setHorizontalAlignment(JLabel.LEFT);
        lblErrorResult.setFont(new Font("Dialog", Font.PLAIN, 14));
        add(lblErrorResult);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        setVisible(true);
    }

    private int getFunction(int idx) {
        if ( idx == 0 ) {
            return BackpropagationThread.SIGMOID_FUNCTION;
        }
        if ( idx == 1 ) {
            return BackpropagationThread.TANH_FUNCTION;
        }
        return BackpropagationThread.RELU_FUNCTION;
    }

    public void setModel(BackpropagationThread.Model model) {
        this.model = model;
        clickEnable = true;
        modelEnable = true;
        addInstanceEnable = false;
        jmPredict.setVisible(true);
        jmiPerceptronPredict.setSelected(true);
        System.out.println("Modelo obtenido");
        System.out.println(model);
        changeUIState(true);
        showSweep();
    }

    public void showSweep() {
        ArrayList<Point> newPoints = new ArrayList<>();
        for ( Point point : points ) {
            if ( !point.sweep ) {
                newPoints.add(point);
            }
        }
        points = newPoints;
        newPoints = new ArrayList<>();
        firstFor:
        for ( int i = 0; i < map.getWidth(); i+= 1 ) {
            for ( int j = 0; j < map.getHeight(); j+= 1 ) {
                Point point = new Point();
                point.xMap = i;
                point.yMap = j;
                point.x = ( i >= MAP_WIDTH * 0.5 ) ? i - ( MAP_WIDTH * 0.5 ) : -((MAP_WIDTH * 0.5) - i);
                point.x /= (MAP_WIDTH * 0.5) / MAP_SCALE;
                point.y = ( j > MAP_HEIGHT * 0.5 ) ? -(j - (MAP_HEIGHT * 0.5)) : (MAP_HEIGHT * 0.5) - j;
                point.y /= (MAP_HEIGHT * 0.5) / MAP_SCALE;
                Object[] instance = new Object[2];
                instance[0] = point.x;
                instance[1] = point.y;
                Object[] result;
                try {
                    result = model.predictWithPercentage(instance);
                    double color = Math.abs((Double) result[1]);
                    if ( color > 1 ) {
                        color = 1;
                    }
                    point.sweep = true;
                    point.target = Integer.parseInt((String) result[0]);
                    point.color = targetColors.get(point.target);
                    int intensity = (int) (color * 255 * .70);
                    point.color = new Color(point.color.getRed(), point.color.getGreen(), point.color.getBlue(), intensity);
                    newPoints.add(point);
                } catch (Exception ex) {
                    ex.printStackTrace();
                    System.out.println("No se pudo realizar el barrido");
                    break firstFor;
                }
            }
        }
        newPoints.addAll(points);
        points = newPoints;
        map.repaint();
    }

    public void updateErrorAndEpoch(boolean stop, boolean done, double error, int epoch, double learningRate) {
        if ( stop ) {
            if ( done ) {
                lblEpochResult.setText("<html>Convergio en la epoca: <b>" + epoch + "</b></html>");
            } else {
                lblEpochResult.setText("<html>No convergio. Epocas: <b>" + epoch + "</b></html>");
            }
        } else {
            lblEpochResult.setText("<html>Epoca: <b>" + epoch + "</b></html>");
        }
        lblErrorResult.setText("<html>Error: <b>" + error + "</b></html>");
        lblLearningRateResult.setText("<html>Learning rate: <b>"+ learningRate +"</b></html>");
    }

    public void changeUIState(boolean enable) {
        jmOptions.setEnabled(enable);
        jcbValuesOfInstances.setEnabled(enable);
        btnAddInstance.setEnabled(enable);
        jrbFirstHiddenLayer.setEnabled(enable);
        if ( jrbFirstHiddenLayer.isSelected() ) {
            txtFHL.setEnabled(enable);
            jcbFirstLayerFunction.setEnabled(enable);
        }
        jrbSecondHiddenLayer.setEnabled(enable);
        if ( jrbSecondHiddenLayer.isSelected() ) {
            txtSHL.setEnabled(enable);
            jcbSecondLayerFunction.setEnabled(enable);
        }
        txtLearningRate.setEnabled(enable);
        txtEpochs.setEnabled(enable);
        txtMinError.setEnabled(enable);
        rbtnBatch.setEnabled(enable);
        rbtnStochastic.setEnabled(enable);
        rbtnMiniBatch.setEnabled(enable);
        if ( rbtnMiniBatch.isSelected() ) {
            txtMiniBatch.setEnabled(enable);
        }
        btnBackpropagation.setEnabled(enable);
        jcbLastLayerFunction.setEnabled(enable);
    }

    public void addErrorForChart(int epoch, double error, boolean last) {
        if ( errorChartBPWindow != null ) {
            errorChartBPWindow.addValueForSeries(epoch, error, last);
        }
    }

    private class Map extends JPanel {

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            // Obtenemos el alto y ancho del componente
            int width = getWidth();
            int height = getHeight();
            // Linea vertical del lienzo
            g.drawLine(width / 2, 0, width / 2, height);
            // Linea horizontal
            g.drawLine(0, height / 2, width, height / 2);
            // ALgunas líneas más de apoyo
            g.setColor(new Color(170, 183, 184));
            for ( int i = 1; i < (width / MAP_SCALE / 10); i++ ) {
                int point = (int) (i * width / MAP_SCALE / 2);
                if ( point == width / 2 )  {
                    continue;
                }
                g.drawLine(point, 0, point, height);
                g.drawLine(0, point, width, point);
            }
            // Dibujamos los puntos hasta ahora obtenidos
            for ( Point point : points ) {
                if ( point.sweep ) {
                    g.setColor(point.color);
                    g.fillRect(point.xMap, point.yMap, 1, 1);
                } else {
                    g.setColor(Color.BLACK);
                    g.fillOval((point.xMap - RADIUS_POINT) - 1, (point.yMap - RADIUS_POINT) - 1, (RADIUS_POINT * 2) + 2, (RADIUS_POINT * 2) + 2);
                    g.setColor(point.color);
                    g.fillOval(point.xMap - RADIUS_POINT, point.yMap - RADIUS_POINT, RADIUS_POINT * 2, RADIUS_POINT * 2);
                }
            }
        }

    }

    private static class Point {

        public int xMap;
        public int yMap;
        public double x;
        public double y;
        public int target;
        public boolean sweep;
        public Color color;

        @Override
        public String toString() {
            return "Point{" +
                    "xMap=" + xMap +
                    ", yMap=" + yMap +
                    ", x=" + x +
                    ", y=" + y +
                    ", target=" + target +
                    ", sweep=" + sweep +
                    ", color=" + color +
                    '}';
        }

    }

    private static class CustomKeyListener extends KeyAdapter {

        private final JTextField txtField;

        public CustomKeyListener(JTextField txtField)
        {
            this.txtField = txtField;
        }

        @Override
        public void keyTyped(KeyEvent e) {
            if ( (e.getKeyChar() < '0' || e.getKeyChar() > '9') && ( e.getKeyChar() != '.' && e.getKeyChar() != '-' ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '-' && ( txtField.getCaretPosition() != 0 || txtField.getText().contains("-") ) ) {
                e.consume();
            }
            if ( e.getKeyChar() == '.' && !txtField.getText().isEmpty() && ( txtField.getText().contains(".") ) ) {
                e.consume();
            }
            if ( txtField.getText().startsWith("-") && txtField.getCaretPosition() == 0 ) {
                e.consume();
            }
            super.keyTyped(e);
        }
    }

}
