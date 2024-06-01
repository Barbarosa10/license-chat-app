//chat-app server
export const host = "http://localhost:5000";
export const loginRoute = `${host}/api/auth/login`;
export const registerRoute = `${host}/api/auth/register`;
export const logoutRoute = `${host}/api/auth/logout`;
export const createConversationRoute = `${host}/api/auth/createconversation`;
export const updateConversationRoute = `${host}/api/auth/updateconversation`;
export const allContactsRoute = `${host}/api/auth/allcontacts`;
export const allConversationsRoute = `${host}/api/auth/allconversations`;
export const contactRoute = `${host}/api/auth/contact`;
export const sendMessageRoute = `${host}/api/messages/addmsg`;
export const receiveMessageRoute = `${host}/api/messages/getmsg`;
export const setAvatarRoute = `${host}/api/auth/setavatar`;

//sentiment-analysis server
export const sentimentAnalysisRoute = "http://localhost:8001/api/process-data";