package utils;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Properties;

/**
 * Class which can be used to convert a .proporties files to a Proporties class
 */
public class propertyLoader {
    /**
     * Load .proporties file and converts it to the Properties class
     *
     * @param path Path of .proporties file
     * @return Proporties class of .properties file
     * @throws IOException
     */
    public static Properties getPropertyValues(String path) throws IOException {
        FileReader reader = new FileReader(new File(path));
        Properties props = new Properties();
        props.load(reader);

        return props;
    }
}