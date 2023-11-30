const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
const socket = require("socket.io");

const authRoutes = require("./routes/auth_users");
const messageRoutes = require("./routes/messages");

const app = express();
const port = 5000;

app.use(cors());
app.use(express.json());

const url = "mongodb+srv://admin:admin@cluster.usyyx.mongodb.net/ChatApp?retryWrites=true&w=majority"
mongoose
  .connect(url)
  .then(() => {
    console.log("DB Connection Successfull");
  })
  .catch((err) => {
    console.log(err.message);
  });

app.use("/api/auth", authRoutes);
app.use("/api/messages", messageRoutes);

app.get('/', (req, res) => {
  res.send('Welcome to my server!');
});

const server = app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

const io = socket(server, {
  cors: {
    origin: "http://localhost:3000",
    credentials: true,
  },
});

global.onlineUsers = new Map();
console.log(onlineUsers['654f9984b467538aadfe94b9'])
io.on("connection", (socket) => {
  global.chatSocket = socket;
  socket.on("add-user", (userId) => {
    console.log(userId);
    // if(onlineUsers.get(userId) == undefined){
      onlineUsers.set(userId, socket.id);
    // }
    console.log(onlineUsers);
  });
  socket.on("send-msg", (data) => {

    const sendUserSocket = onlineUsers.get(data.to);
    console.log(data.to);
    console.log(sendUserSocket);
    if (sendUserSocket) {
      console.log(data);
      socket.to(sendUserSocket).emit("msg-receive", data);
    }
  });
});
