<%@ page language="java" contentType="text/html; charset=ISO-8859-1"%>
<%@ page import="javax.servlet.http.*, javax.servlet.*" %>
<%
    // Retrieve the counter value from session or set to 0 if not present
    Integer counter = (Integer) session.getAttribute("counter");
    if (counter == null) {
        counter = 0; // Initial counter value
    }

    // Get the action parameter to check whether the user clicked "next" or "prev"
    String action = request.getParameter("action");
    if (action != null) {
        if (action.equals("next")) {
            counter++;
        } else if (action.equals("prev")) {
            counter--;
        }
        session.setAttribute("counter", counter); // Store updated counter in session
        
        // Optionally, set the action to null after it's processed
        action = null; // Not needed, but you can clear it if you prefer
    }
%>
<!DOCTYPE html>
<html>
<head>
    <meta charset="ISO-8859-1">
    <title>Counter Application</title>
</head>
<body>
    <h1>Counter: <%= counter %></h1>

    <!-- 1. Hidden Field Method -->
    <form method="post" action="counter.jsp">
        <input type="hidden" name="counter" value="<%= counter %>">
        <input type="submit" name="action" value="prev">
        <input type="submit" name="action" value="next">
    </form>

    <br>
    <!-- 2. URL Rewriting Method -->
    <a href="counter.jsp?action=prev&counter=<%= counter - 1 %>">Prev</a>
    <a href="counter.jsp?action=next&counter=<%= counter + 1 %>">Next</a>

    <br>
    <!-- 3. Cookies Method -->
    <%
        Cookie counterCookie = new Cookie("counter", counter.toString());
        counterCookie.setMaxAge(60*60*24); // Store cookie for 1 day
        response.addCookie(counterCookie);
    %>
    <a href="counter.jsp?action=prev">Prev (Cookie)</a>
    <a href="counter.jsp?action=next">Next (Cookie)</a>

    <br>
    <!-- 4. Session API -->
    <a href="counter.jsp?action=prev">Prev (Session API)</a>
    <a href="counter.jsp?action=next">Next (Session API)</a>

</body>
</html>
