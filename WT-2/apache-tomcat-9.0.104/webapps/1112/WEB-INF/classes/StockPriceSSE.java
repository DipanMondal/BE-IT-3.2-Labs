import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/stockprices")
public class StockPriceSSE extends HttpServlet {

    private Random random = new Random();

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        response.setContentType("text/event-stream");
        response.setCharacterEncoding("UTF-8");

        PrintWriter out = response.getWriter();

        while (true) {
            try {
                // Generate random stock prices
                double stockA = 100 + random.nextDouble() * 10;  // 100 - 110
                double stockB = 200 + random.nextDouble() * 20;  // 200 - 220

                // Create JSON string
                String json = String.format("{\"stockA\": \"%.2f\", \"stockB\": \"%.2f\"}", stockA, stockB);

                // Send SSE event
                out.write("data: " + json + "\n\n");
                out.flush();

                Thread.sleep(2000); // update every 2 seconds

            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
