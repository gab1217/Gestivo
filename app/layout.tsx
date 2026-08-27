import type { Metadata } from "next";
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

export function generateMetadata(): Metadata {
  const origin = "https://gab1217.github.io/Gestivo";
  const title = "Gestivo — Filipino Sign Language Recognition App";
  const description = "Gestivo recognizes Filipino Sign Language letters and converts them into real-time text and speech on web, Android, and Windows.";
  const socialImage = `${origin}/og-v2.png`;

  return {
    title,
    description,
    icons: { icon: "/gestivo-logo.png", shortcut: "/gestivo-logo.png" },
    manifest: "/manifest.webmanifest",
    applicationName: "Gestivo",
    metadataBase: new URL(`${origin}/`),
    alternates: { canonical: `${origin}/` },
    robots: { index: true, follow: true, googleBot: { index: true, follow: true, "max-image-preview": "large" } },
    keywords: ["Gestivo", "Filipino Sign Language", "FSL recognizer", "sign language app", "FSL translator"],
    appleWebApp: { capable: true, title: "Gestivo", statusBarStyle: "black-translucent" },
    openGraph: {
      title,
      description,
      type: "website",
      url: `${origin}/`,
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
