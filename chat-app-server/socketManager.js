const socket = require("socket.io");

const users = new Map();
const onlineUsers = new Map();

const initSocket = (server) => {

  const io = socket(server, {
    cors: {
      origin: process.env.ORIGIN_CORS_URL,
      credentials: true,
    },
  });

  io.on("connection", (socket) => {
    const userId = socket.handshake.query.userId;

    socket.on("add-user", (userId) => {
      onlineUsers.set(userId, socket.id);
    });

    socket.on("add-user-video", (userId) => {
      users.set(userId, socket.id);
    });

    socket.on("send-msg", (data) => {
      const sendUserSocket = onlineUsers.get(data.to);
      if (sendUserSocket) {
        socket.to(sendUserSocket).emit("msg-receive", data);
      }
    });

    socket.on("open-socket", (data) => {
      const sendUserSocket = onlineUsers.get(data.to);
      if (sendUserSocket) {
        socket.to(sendUserSocket).emit("open-socket", { chatId: data.chatId, to: data.to, from: data.from });
      }
    });

    socket.on("socket-opened", (data) => {
      const sendUserSocket = onlineUsers.get(data.to);
      if (sendUserSocket) {
        socket.to(sendUserSocket).emit("socket-opened", data);
      }
    });

    socket.on("end-call", (data) => {
      const sendUserSocket = users.get(data.to);
      if (sendUserSocket) {
        socket.to(sendUserSocket).emit("callEnded", data);
      }
    });

    socket.on("disconnect", () => {
      socket.broadcast.emit("disconnected");
    });

    socket.on("callUser", (data) => {
      const sendUserSocket = users.get(data.userToCall);
      socket.to(sendUserSocket).emit("callUser", { chatId: data.chatId, lastMessage: data.lastMessage, timestamp: data.timestamp, signal: data.signalData, from: data.from, username: data.username });
    });

    socket.on("answerCall", (data) => {
      const sendUserSocket = users.get(data.to);
      socket.to(sendUserSocket).emit("callAccepted", data.signal);
    });

  });

};

module.exports = { initSocket, onlineUsers };
