import java.io.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;
import java.sql.Connection;
import java.sql.DriverManager;

class JDBCUtil {
    public static Connection getConnection() throws Exception {
        Class.forName("com.mysql.jdbc.Driver");
        return DriverManager.getConnection(
            "jdbc:mysql://172.16.4.234:3306/test", "guest", "guest"
        );
    }
}

public class SearchByName extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
    throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();

        String searchName = request.getParameter("searchName");

        try (Connection con = JDBCUtil.getConnection()) {
            PreparedStatement pst = con.prepareStatement(
                "SELECT * FROM students112 WHERE name LIKE ?");
            pst.setString(1, "%" + searchName + "%");
            ResultSet rs = pst.executeQuery();

            out.println("<h2>Students Matching \"" + searchName + "\":</h2><ul>");
            while (rs.next()) {
                out.println("<li>" + rs.getInt("roll_no") + " - " +
                            rs.getString("name") + " - " +
                            rs.getString("dept_name") + "</li>");
            }
            out.println("</ul>");
        } catch (Exception e) {
            e.printStackTrace(out);
        }
    }
}
