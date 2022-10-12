package tests;

import com.formdev.flatlaf.FlatIntelliJLaf;
import windows.BackpropagationWindow;

import javax.swing.*;

public class BackpropagationTest {

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel( new FlatIntelliJLaf() );
        } catch (Exception ex) {
            System.out.println("Look and feer error!");
        }
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new BackpropagationWindow();
            }
        });
    }

}
