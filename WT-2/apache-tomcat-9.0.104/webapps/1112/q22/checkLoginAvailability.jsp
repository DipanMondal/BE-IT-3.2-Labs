<%@ page import="java.sql.*, javax.sql.*, java.util.*" %>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1" pageEncoding="ISO-8859-1"%>

<%
    String loginName = request.getParameter("login_name");
    Connection conn = null;
    PreparedStatement stmt = null;
    ResultSet rs = null;

    String dbURL = "jdbc:mysql://172.16.4.234:3306/test";
    String dbUsername = "guest";  // Replace with your MySQL username
    String dbPassword = "guest";  // Replace with your MySQL password

    try {
        // Set up the database connection
        Class.forName("com.mysql.jdbc.Driver");
        conn = DriverManager.getConnection(dbURL, dbUsername, dbPassword);

        // Query to check if login name already exists
        String query = "SELECT * FROM users1 WHERE login_name = ?";
        stmt = conn.prepareStatement(query);
        stmt.setString(1, loginName);

        rs = stmt.executeQuery();

        if (rs.next()) {
            out.print("Login name is already taken!");
        } else {
            out.print("Login name is available.");
        }
    } catch (Exception e) {
        e.printStackTrace();
    } finally {
        try {
            if (rs != null) rs.close();
            if (stmt != null) stmt.close();
            if (conn != null) conn.close();
        } catch (SQLException se) {
            se.printStackTrace();
        }
    }
%>
