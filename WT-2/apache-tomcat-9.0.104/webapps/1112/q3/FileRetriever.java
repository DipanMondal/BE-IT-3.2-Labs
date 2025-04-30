import java.io.*;
import java.net.Socket;

class FileRetriever {
    public static void main(String[] args) {
        String serverAddress = "172.16.4.228"; // Replace with the remote machine's IP
        int port = 8080; // Default Tomcat port

        try (Socket socket = new Socket(serverAddress, port);
             PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
             BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()))) {

            // Send HTTP GET request
            out.println("GET /1112/ass1/q1/first.html HTTP/1.1");
            out.println("Host: " + serverAddress);
            out.println("Connection: close");
            out.println(); // Blank line to indicate end of headers

            // Read the response
            String line;
            boolean isContent = false;
            StringBuilder content = new StringBuilder();

            while ((line = in.readLine()) != null) {
                if (isContent) {
                    content.append(line).append("\n");
                }
                // HTTP headers and blank line separate the content
                if (line.isEmpty()) {
                    isContent = true;
                }
            }

            // Save content to a local file
            try (BufferedWriter writer = new BufferedWriter(new FileWriter("first.html"))) {
                writer.write(content.toString());
            }

            System.out.println("File retrieved and saved as 'first.html'.");

        } catch (IOException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}