import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.util.*;

public class Subjects extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        // Setting the response content type
        response.setContentType("text/html");

        // Fetching the selected year and semester from the request
        String year = request.getParameter("year");
        String sem = request.getParameter("sem");

        // Mapping subjects for each year and semester
        Map<String, String[]> subjectsMap = new HashMap<>();
        subjectsMap.put("1-1", new String[]{"Mathematics I", "Physics I", "Programming Fundamentals"});
        subjectsMap.put("1-2", new String[]{"Mathematics II", "Chemistry", "Data Structures"});
        subjectsMap.put("2-1", new String[]{"Discrete Mathematics", "Database Systems", "Operating Systems"});
        subjectsMap.put("2-2", new String[]{"Algorithm Design", "Computer Networks", "Software Engineering"});
        subjectsMap.put("3-1", new String[]{"Artificial Intelligence", "Compiler Design", "Machine Learning"});
        subjectsMap.put("3-2", new String[]{"Web Technology", "Cloud Computing", "Mobile Computing"});
        subjectsMap.put("4-1", new String[]{"Big Data Analytics", "Cybersecurity", "IoT"});
        subjectsMap.put("4-2", new String[]{"Project Work", "Electives", "Internship"});

        // Creating the key for mapping
        String key = year + "-" + sem;

        // Retrieving subjects based on year and semester
        String[] subjects = subjectsMap.getOrDefault(key, new String[]{"No subjects available for the selected year and semester."});

        // Writing the response HTML
        PrintWriter out = response.getWriter();
        out.println("<html>");
        out.println("<head><title>Subjects List</title></head>");
        out.println("<body>");
        out.println("<h1>Subjects for Year " + year + " and Semester " + sem + "</h1>");
        out.println("<ul>");
        for (String subject : subjects) {
            out.println("<li>" + subject + "</li>");
        }
        out.println("</ul>");
        out.println("<a href='/1112/q8/index.html'>Go Back</a>"); // Link to go back to the form
        out.println("</body>");
        out.println("</html>");
    }
}


/* 
javac -cp "C:\Program Files\Apache Software Foundation\Tomcat 9.0\lib\servlet-api.jar" Subjects.java
*/
