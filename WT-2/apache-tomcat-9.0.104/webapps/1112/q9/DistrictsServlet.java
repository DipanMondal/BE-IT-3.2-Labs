import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.util.*;

public class DistrictsServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String state = request.getParameter("state");
        String districts = "";

        // Map of states and their corresponding districts
        Map<String, List<String>> stateDistrictMap = new HashMap<>();
        stateDistrictMap.put("West Bengal", Arrays.asList("Kolkata", "Darjeeling", "Siliguri", "Howrah"));
        stateDistrictMap.put("Maharashtra", Arrays.asList("Mumbai", "Pune", "Nagpur", "Aurangabad"));
        stateDistrictMap.put("Karnataka", Arrays.asList("Bengaluru", "Mysuru", "Mangaluru", "Hubli"));
        stateDistrictMap.put("Tamil Nadu", Arrays.asList("Chennai", "Coimbatore", "Madurai", "Salem"));
        stateDistrictMap.put("Uttar Pradesh", Arrays.asList("Lucknow", "Kanpur", "Varanasi", "Agra"));

        if (stateDistrictMap.containsKey(state)) {
            districts = String.join(",", stateDistrictMap.get(state));
        }

        // Send the district list as a plain response
        response.setContentType("text/plain");
        PrintWriter out = response.getWriter();
        out.print(districts);
    }
}
