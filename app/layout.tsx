import type { Metadata } from "next";
import { headers } from "next/headers";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const incomingHeaders = await headers();
  const host = incomingHeaders.get("x-forwarded-host") ?? incomingHeaders.get("host") ?? "localhost:3000";
  const protocol = incomingHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;
  const title = "Gestivo — Every gesture finds its voice";
  const description = "A Filipino Sign Language assistant for real-time text and speech, powered privately in your browser.";
  const socialImage = `${origin}/og-v2.png`;

  return {
    title,
    description,
    icons: { icon: "/gestivo-logo.png", shortcut: "/gestivo-logo.png" },
    manifest: "/manifest.webmanifest",
    applicationName: "Gestivo",
    appleWebApp: { capable: true, title: "Gestivo", statusBarStyle: "black-translucent" },
    openGraph: {
      title,
      description,
      type: "website",
      url: origin,
      images: [{ url: socialImage, width: 1200, height: 630, alt: "Gestivo — Every gesture finds its voice" }],
    },
    twitter: { card: "summary_large_image", title, description, images: [socialImage] },
  };
}

export const viewport = {
  themeColor: "#071510",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
