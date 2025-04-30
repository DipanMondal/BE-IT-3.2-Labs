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

public class SearchByDepartment extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response)
    throws ServletException, IOException {
        response.setContentType("text/html");
        PrintWriter out = response.getWriter();

        String deptName = request.getParameter("deptName");

        try (Connection con = JDBCUtil.getConnection()) {
            PreparedStatement pst = con.prepareStatement(
                "SELECT * FROM students112 WHERE dept_name = ?");
            pst.setString(1, deptName);
            ResultSet rs = pst.executeQuery();

            out.println("<h2>Students in " + deptName + " Department:</h2><ul>");
            while (rs.next()) {
                out.println("<li>" + rs.getInt("roll_no") + " - " +
                            rs.getString("name") + "</li>");
            }
            out.println("</ul>");
        } catch (Exception e) {
            e.printStackTrace(out);
        }
    }
}
