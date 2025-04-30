import javax.websocket.OnClose;
import javax.websocket.OnError;
import javax.websocket.OnMessage;
import javax.websocket.OnOpen;
import javax.websocket.Session;
import javax.websocket.server.ServerEndpoint;
import java.io.IOException;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

@ServerEndpoint("/chat")
public class ChatServerEndpoint {

    private static Set<Session> users = Collections.synchronizedSet(new HashSet<Session>());

    @OnOpen
    public void onOpen(Session session) {
        users.add(session);
    }

    @OnMessage
    public void onMessage(String message, Session session) throws IOException {
        for (Session user : users) {
            if (user.isOpen()) {
                user.getBasicRemote().sendText(message);
            }
        }
    }

    @OnClose
    public void onClose(Session session) {
        users.remove(session);
    }

    @OnError
    public void onError(Session session, Throwable throwable) {
        users.remove(session);
        throwable.printStackTrace();
    }
}
