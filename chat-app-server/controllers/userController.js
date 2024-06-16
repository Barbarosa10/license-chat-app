const User = require("../models/userModel");
const Conversation = require("../models/conversationModel");
const jwt = require('jsonwebtoken');
const bcrypt = require("bcrypt");
const { onlineUsers } = require("../socketManager");


module.exports.login = async (req, res, next) => {
  try {
    const { username, password } = req.body;
    const user = await User.findOne({ username });

    onlineUsers.get(user._id.toString())

    if(onlineUsers.get(user._id.toString())){
      return res.status(400).json({msg: "User logged from another device!", status: false});
    }

    if (!user)
      return res.status(400).json({ msg: "Incorrect credentials", status: false });

    const isPasswordValid = await bcrypt.compare(password, user.password);

    if (!isPasswordValid)
      return res.status(400).json({ msg: "Incorrect credentials", status: false });

    const token = jwt.sign({ id: user._id , username: user.username, email: user.email}, process.env.JWT_SECRET, { expiresIn: '1h' });

    return res.json({ status: true, token, _id: user._id, email: user.email, username: user.username, avatarImage: user.avatarImage });
  } catch (ex) {
    next(ex);
  }
};

module.exports.register = async (req, res, next) => {
  try {
    const { username, email, password, avatarImage} = req.body;

    const hashedPassword = await bcrypt.hash(password, 10);

    const usernameCheck = await User.findOne({ username });

    if (usernameCheck)
      return res.status(400).json({ msg: "Username already used", status: false });

    const emailCheck = await User.findOne({ email });
    if (emailCheck)
      return res.status(400).json({ msg: "Email already used", status: false });

    if(avatarImage === undefined || avatarImage === ""){
      return res.status(400).json({ msg: "Select a valid avatar image", status: false });
    }

    const user = await User.create({
      email,
      username,
      password: hashedPassword,
      avatarImage
    });

    delete user.password;
    return res.json({ status: true, user });
  } catch (ex) {
    next(ex);
  }
};

module.exports.getAllContacts = async (req, res, next) => {
  try {
    const users = await User.find({ _id: { $ne: req.params.id } }).select([
      "email",
      "username",
      "avatarImage",
      "_id",
    ]);
    return res.json(users);
  } catch (ex) {
    next(ex);
  }
};

module.exports.getAllConversations = async (req, res, next) => {
  try {
    const conversations = await Conversation.find({ 
      participants: { $in: [req.params.username] } 
    }).select([
      "participants",
      "lastMessage",
      "_id",
      "timestamp"
    ]);
    return res.json(conversations);
  } catch (ex) {
    next(ex);
  }
};

module.exports.getContact = async (req, res, next) => {
  try {
    const user = await User.find({ username: req.params.username  }).select([
      "avatarImage",
      "_id",
      "username"
    ]);
    return res.json(user);
  } catch (ex) {
    next(ex);
  }
};


module.exports.createConversation = async (req, res, next) => {
  const {participants} = req.body;
  const lastMessage = "";
  try {
    const newConversation = new Conversation({
      participants,
      lastMessage
    });
    // console.log(newConversation);
    const savedConversation = await newConversation.save();
    return res.json(savedConversation);
  } catch (ex) {
    next(ex);
  }
};

module.exports.updateLastMessage = async (req, res, next) => {
  const {lastMessage, conversationId} = req.body;

  try {
    const updatedLastMessage = {
      $set: {
        lastMessage,
        timestamp: Date.now(),
      },
    };

    const result = await Conversation.updateOne({_id: conversationId}, updatedLastMessage)
    return res.json(result);
  } catch (ex) {
    next(ex);
  }
};

module.exports.logOut = (req, res, next) => {
  try {
    if (!req.params.id) return res.json({ msg: "User id is required " });
    onlineUsers.delete(req.params.id);
    return res.status(200).send();
  } catch (ex) {
    next(ex);
  }
};
