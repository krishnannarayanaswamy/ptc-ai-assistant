import { GeistSans } from "geist/font/sans";
import "./globals.css";

export const metadata = {
  title: "PTC AI Assistant",
  description: "PTC AI Assistant - Powered by AI with DataStax",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={GeistSans.variable}>
      <body>{children}</body>
    </html>
  );
}
