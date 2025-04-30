import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class DistrictInfoServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String state = request.getParameter("state");
        String district = request.getParameter("district");

        // Sample district information
        String info = "No information available.";
        if ("Kolkata".equals(district)) {
            info = "Kolkata: Capital of West Bengal, known as the 'City of Joy'.";
        } else if ("Mumbai".equals(district)) {
            info = "Mumbai: Financial capital of India, home to Bollywood.";
        } else if ("Bengaluru".equals(district)) {
            info = "Bengaluru: Capital of Karnataka, known as the 'Silicon Valley of India'.";
        } else if ("Chennai".equals(district)) {
            info = "Chennai: Capital of Tamil Nadu, known for its cultural heritage and IT hubs.";
        } else if ("Lucknow".equals(district)) {
            info = "Lucknow: Capital of Uttar Pradesh, known as the 'City of Nawabs'.";
        }

        // Write response
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();
        out.println("<html><head><title>District Information</title></head>");
        out.println("<body>");
        out.println("<h1>Information about " + district + ", " + state + "</h1>");
        out.println("<p>" + info + "</p>");
        out.println("<a href='/1112/q9/index.html'>Go Back</a>");
        out.println("</body></html>");
    }
}
